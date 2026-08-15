"""Immutable two-setting predictive worker and panel plans for M03R-v15."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from rl_quant.protocol.hold30_alpha_m03r_v15_top2000_dev import (
    M03R_V15_DESIGN_ID,
    M03R_V15_PREDICTIVE_SPEC,
    M03R_V15_PROTOCOL_GENERATION,
    M03R_V15_PROTOCOL_SHA256,
    M03R_V15_SELECTED_HORIZON_SESSIONS,
    M03R_V15_SETTING_IDS,
    resolve_m03r_v15_setting,
)
from rl_quant.training.top2000_m03r_v15_fold import render_m03r_v15_fold_geometries

M03R_V15_WORKER_PLAN_SCHEMA = "rl-quant.top2000-dev.m03r-v15-worker-plan-v1"
M03R_V15_PANEL_PLAN_SCHEMA = "rl-quant.top2000-dev.m03r-v15-panel-plan-v1"
M03R_V15_FOLD_UPDATE_COUNTS = tuple(
    geometry.optimizer_updates for geometry in render_m03r_v15_fold_geometries(1001)
)


class M03RV15PredictiveWorkerError(ValueError):
    """The v15 worker or paired panel identity drifted."""


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
        raise M03RV15PredictiveWorkerError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class M03RV15PredictiveWorkerPlan:
    setting_index: int
    setting_id: str
    output_root: str
    cache_path: str
    initial_parameter_state_path: str
    panel_episode_schedule_sha256: str
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
    structural_preflight_path: str
    structural_preflight_file_sha256: str
    structural_preflight_receipt_sha256: str
    fold_optimizer_updates: tuple[int, ...] = M03R_V15_FOLD_UPDATE_COUNTS
    seed: int = M03R_V15_PREDICTIVE_SPEC.seed
    fold_count: int = M03R_V15_PREDICTIVE_SPEC.chronological_fold_count
    expected_world_size: int = M03R_V15_PREDICTIVE_SPEC.expected_world_size
    h100s_per_worker: int = 2
    economic_optimizer_updates: int = 0
    selected_horizon_sessions: int = M03R_V15_SELECTED_HORIZON_SESSIONS
    checkpoint_round_trip_required: bool = True
    full_context_required: bool = True
    every_origin_once_per_epoch_required: bool = True
    separate_target_action_operator_required: bool = True
    package_authorized: bool = False
    kubernetes_launch_authorized: bool = False
    outer_2026_access_authorized: bool = False
    protocol_generation: str = M03R_V15_PROTOCOL_GENERATION
    design_id: str = M03R_V15_DESIGN_ID
    protocol_sha256: str = M03R_V15_PROTOCOL_SHA256
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    schema: str = M03R_V15_WORKER_PLAN_SCHEMA

    def validate(self) -> None:
        setting = resolve_m03r_v15_setting(self.setting_index)
        for name in (
            "panel_episode_schedule_sha256",
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
            "structural_preflight_file_sha256",
            "structural_preflight_receipt_sha256",
        ):
            _digest(name, getattr(self, name))
        if (
            self.setting_id != setting.setting_id
            or not self.output_root
            or not self.cache_path
            or not self.initial_parameter_state_path
            or not self.risk_source_manifest_path
            or not self.projector_manifest_path
            or not self.structural_preflight_path
            or self.fold_optimizer_updates != M03R_V15_FOLD_UPDATE_COUNTS
            or self.seed != M03R_V15_PREDICTIVE_SPEC.seed
            or self.fold_count != M03R_V15_PREDICTIVE_SPEC.chronological_fold_count
            or self.expected_world_size
            != M03R_V15_PREDICTIVE_SPEC.expected_world_size
            or self.h100s_per_worker != 2
            or self.economic_optimizer_updates != 0
            or self.selected_horizon_sessions
            != M03R_V15_SELECTED_HORIZON_SESSIONS
            or not self.checkpoint_round_trip_required
            or not self.full_context_required
            or not self.every_origin_once_per_epoch_required
            or not self.separate_target_action_operator_required
            or self.package_authorized
            or self.kubernetes_launch_authorized
            or self.outer_2026_access_authorized
            or self.protocol_generation != M03R_V15_PROTOCOL_GENERATION
            or self.design_id != M03R_V15_DESIGN_ID
            or self.protocol_sha256 != M03R_V15_PROTOCOL_SHA256
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.schema != M03R_V15_WORKER_PLAN_SCHEMA
        ):
            raise M03RV15PredictiveWorkerError("v15 worker plan drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class M03RV15PredictivePanelPlan:
    workers: tuple[M03RV15PredictiveWorkerPlan, ...]
    indexed_completions: int = 2
    parallelism: int = 2
    h100s_per_completion: int = 2
    maximum_h100_requests: int = M03R_V15_PREDICTIVE_SPEC.maximum_h100_requests
    economic_panel_authorized: bool = False
    package_authorized: bool = False
    kubernetes_launch_authorized: bool = False
    protocol_sha256: str = M03R_V15_PROTOCOL_SHA256
    schema: str = M03R_V15_PANEL_PLAN_SCHEMA

    def validate(self) -> None:
        for worker in self.workers:
            worker.validate()
        shared_fields = (
            "panel_episode_schedule_sha256",
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
            "structural_preflight_path",
            "structural_preflight_file_sha256",
            "structural_preflight_receipt_sha256",
            "fold_optimizer_updates",
            "selected_horizon_sessions",
        )
        if (
            len(self.workers) != len(M03R_V15_SETTING_IDS)
            or tuple(worker.setting_index for worker in self.workers) != (0, 1)
            or tuple(worker.setting_id for worker in self.workers)
            != M03R_V15_SETTING_IDS
            or any(
                len({getattr(worker, field) for worker in self.workers}) != 1
                for field in shared_fields
            )
            or len({worker.output_root for worker in self.workers}) != len(self.workers)
            or len({worker.receipt_sha256 for worker in self.workers})
            != len(self.workers)
            or self.indexed_completions != 2
            or self.parallelism != 2
            or self.h100s_per_completion != 2
            or self.maximum_h100_requests != 4
            or self.parallelism * self.h100s_per_completion
            != self.maximum_h100_requests
            or self.economic_panel_authorized
            or self.package_authorized
            or self.kubernetes_launch_authorized
            or self.protocol_sha256 != M03R_V15_PROTOCOL_SHA256
            or self.schema != M03R_V15_PANEL_PLAN_SCHEMA
        ):
            raise M03RV15PredictiveWorkerError("v15 panel plan drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


__all__ = [
    "M03R_V15_FOLD_UPDATE_COUNTS",
    "M03R_V15_PANEL_PLAN_SCHEMA",
    "M03R_V15_WORKER_PLAN_SCHEMA",
    "M03RV15PredictivePanelPlan",
    "M03RV15PredictiveWorkerError",
    "M03RV15PredictiveWorkerPlan",
]
