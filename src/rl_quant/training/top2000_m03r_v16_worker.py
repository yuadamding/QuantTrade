"""Immutable three-setting predictive worker and panel plans for M03R-v16."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import (
    M03R_V16_DESIGN_ID,
    M03R_V16_PREDICTIVE_SPEC,
    M03R_V16_PROTOCOL_GENERATION,
    M03R_V16_PROTOCOL_SHA256,
    M03R_V16_SETTING_IDS,
    resolve_m03r_v16_setting,
)
from rl_quant.protocol.hold_target import LEGACY_HOLD30_TARGET_SPEC
from rl_quant.training.top2000_m03r_v16_fold import (
    render_m03r_v16_fold_geometries,
)

M03R_V16_WORKER_PLAN_SCHEMA = "rl-quant.top2000-dev.m03r-v16-worker-plan-v1"
M03R_V16_PANEL_PLAN_SCHEMA = "rl-quant.top2000-dev.m03r-v16-panel-plan-v1"
M03R_V16_FOLD_UPDATE_COUNTS = tuple(
    geometry.maximum_optimizer_updates
    for geometry in render_m03r_v16_fold_geometries(1001)
)


class M03RV16PredictiveWorkerError(ValueError):
    """The V16 worker or paired panel identity drifted."""


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _digest(name: str, value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise M03RV16PredictiveWorkerError(
            f"{name} must be a lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, slots=True)
class M03RV16PredictiveWorkerPlan:
    setting_index: int
    setting_id: str
    output_root: str
    cache_path: str
    initial_parameter_state_path: str
    structural_slab_path: str
    panel_schedule_sha256: str
    initial_parameter_state_file_sha256: str
    initial_parameter_state_sha256: str
    initial_parameter_architecture_sha256: str
    cache_sha256: str
    risk_source_manifest_path: str
    risk_source_manifest_file_sha256: str
    projector_manifest_path: str
    projector_manifest_file_sha256: str
    projector_manifest_sha256: str
    projector_binding_sha256: str
    source_manifest_sha256: str
    source_archive_sha256: str
    structural_slab_file_sha256: str
    structural_slab_receipt_sha256: str
    structural_action_operator_root_sha256: str
    structural_target_operator_root_sha256: str
    structural_target_root_sha256: tuple[str, ...]
    fold_optimizer_updates: tuple[int, ...] = M03R_V16_FOLD_UPDATE_COUNTS
    seed: int = M03R_V16_PREDICTIVE_SPEC.seed
    fold_count: int = M03R_V16_PREDICTIVE_SPEC.chronological_fold_count
    expected_world_size: int = M03R_V16_PREDICTIVE_SPEC.expected_world_size
    h100s_per_worker: int = 2
    score_training_epochs: int = M03R_V16_PREDICTIVE_SPEC.score_training_epochs
    hold_target_sessions: int = LEGACY_HOLD30_TARGET_SPEC.target_sessions
    hold_target_spec_sha256: str = LEGACY_HOLD30_TARGET_SPEC.receipt_sha256
    economic_optimizer_updates: int = 0
    reinforcement_learning_updates: int = 0
    fixed_terminal_checkpoint_required: bool = True
    checkpoint_round_trip_required: bool = True
    common_target_support_required: bool = True
    action_projected_score_required: bool = True
    package_authorized: bool = False
    kubernetes_launch_authorized: bool = False
    outer_2026_access_authorized: bool = False
    protocol_generation: str = M03R_V16_PROTOCOL_GENERATION
    design_id: str = M03R_V16_DESIGN_ID
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    schema: str = M03R_V16_WORKER_PLAN_SCHEMA

    def validate(self) -> None:
        setting = resolve_m03r_v16_setting(self.setting_index)
        for name in (
            "panel_schedule_sha256",
            "initial_parameter_state_file_sha256",
            "initial_parameter_state_sha256",
            "initial_parameter_architecture_sha256",
            "cache_sha256",
            "risk_source_manifest_file_sha256",
            "projector_manifest_file_sha256",
            "projector_manifest_sha256",
            "projector_binding_sha256",
            "source_manifest_sha256",
            "source_archive_sha256",
            "structural_slab_file_sha256",
            "structural_slab_receipt_sha256",
            "structural_action_operator_root_sha256",
            "structural_target_operator_root_sha256",
            "hold_target_spec_sha256",
        ):
            _digest(name, getattr(self, name))
        for index, value in enumerate(self.structural_target_root_sha256):
            _digest(f"structural_target_root_sha256[{index}]", value)
        if (
            self.setting_id != setting.setting_id
            or not self.output_root
            or not self.cache_path
            or not self.initial_parameter_state_path
            or not self.structural_slab_path
            or not self.risk_source_manifest_path
            or not self.projector_manifest_path
            or len(self.structural_target_root_sha256) != len(M03R_V16_SETTING_IDS)
            or self.fold_optimizer_updates != M03R_V16_FOLD_UPDATE_COUNTS
            or self.seed != M03R_V16_PREDICTIVE_SPEC.seed
            or self.fold_count != M03R_V16_PREDICTIVE_SPEC.chronological_fold_count
            or self.expected_world_size
            != M03R_V16_PREDICTIVE_SPEC.expected_world_size
            or self.h100s_per_worker != 2
            or self.score_training_epochs
            != M03R_V16_PREDICTIVE_SPEC.score_training_epochs
            or self.hold_target_sessions
            != LEGACY_HOLD30_TARGET_SPEC.target_sessions
            or self.hold_target_spec_sha256
            != LEGACY_HOLD30_TARGET_SPEC.receipt_sha256
            or self.economic_optimizer_updates != 0
            or self.reinforcement_learning_updates != 0
            or not self.fixed_terminal_checkpoint_required
            or not self.checkpoint_round_trip_required
            or not self.common_target_support_required
            or not self.action_projected_score_required
            or self.package_authorized
            or self.kubernetes_launch_authorized
            or self.outer_2026_access_authorized
            or self.protocol_generation != M03R_V16_PROTOCOL_GENERATION
            or self.design_id != M03R_V16_DESIGN_ID
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.schema != M03R_V16_WORKER_PLAN_SCHEMA
        ):
            raise M03RV16PredictiveWorkerError("V16 worker plan drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class M03RV16PredictivePanelPlan:
    workers: tuple[M03RV16PredictiveWorkerPlan, ...]
    indexed_completions: int = 3
    parallelism: int = 3
    h100s_per_completion: int = 2
    maximum_h100_requests: int = M03R_V16_PREDICTIVE_SPEC.maximum_h100_requests
    primary_setting_index: int = M03R_V16_PREDICTIVE_SPEC.primary_setting_index
    economic_panel_authorized: bool = False
    package_authorized: bool = False
    kubernetes_launch_authorized: bool = False
    protocol_sha256: str = M03R_V16_PROTOCOL_SHA256
    schema: str = M03R_V16_PANEL_PLAN_SCHEMA

    def validate(self) -> None:
        for worker in self.workers:
            worker.validate()
        shared_fields = (
            "panel_schedule_sha256",
            "initial_parameter_state_sha256",
            "initial_parameter_state_file_sha256",
            "initial_parameter_architecture_sha256",
            "initial_parameter_state_path",
            "cache_sha256",
            "cache_path",
            "risk_source_manifest_path",
            "risk_source_manifest_file_sha256",
            "projector_manifest_path",
            "projector_manifest_file_sha256",
            "projector_manifest_sha256",
            "projector_binding_sha256",
            "source_manifest_sha256",
            "source_archive_sha256",
            "structural_slab_path",
            "structural_slab_file_sha256",
            "structural_slab_receipt_sha256",
            "structural_action_operator_root_sha256",
            "structural_target_operator_root_sha256",
            "structural_target_root_sha256",
            "fold_optimizer_updates",
            "hold_target_spec_sha256",
        )
        expected_indices = tuple(range(len(M03R_V16_SETTING_IDS)))
        if (
            len(self.workers) != len(M03R_V16_SETTING_IDS)
            or tuple(worker.setting_index for worker in self.workers)
            != expected_indices
            or tuple(worker.setting_id for worker in self.workers)
            != M03R_V16_SETTING_IDS
            or any(
                len({getattr(worker, name) for worker in self.workers}) != 1
                for name in shared_fields
            )
            or len({worker.output_root for worker in self.workers})
            != len(self.workers)
            or len({worker.receipt_sha256 for worker in self.workers})
            != len(self.workers)
            or self.indexed_completions != 3
            or self.parallelism != 3
            or self.h100s_per_completion != 2
            or self.maximum_h100_requests != 6
            or self.parallelism * self.h100s_per_completion
            != self.maximum_h100_requests
            or self.primary_setting_index != 2
            or self.economic_panel_authorized
            or self.package_authorized
            or self.kubernetes_launch_authorized
            or self.protocol_sha256 != M03R_V16_PROTOCOL_SHA256
            or self.schema != M03R_V16_PANEL_PLAN_SCHEMA
        ):
            raise M03RV16PredictiveWorkerError("V16 panel plan drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


__all__ = [
    "M03R_V16_FOLD_UPDATE_COUNTS",
    "M03R_V16_PANEL_PLAN_SCHEMA",
    "M03R_V16_WORKER_PLAN_SCHEMA",
    "M03RV16PredictivePanelPlan",
    "M03RV16PredictiveWorkerError",
    "M03RV16PredictiveWorkerPlan",
]
