"""Immutable paired three-setting worker plan for M03R-v12."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_DESIGN_ID,
    M03R_V12_PREDICTIVE_SPEC,
    M03R_V12_PROTOCOL_GENERATION,
    M03R_V12_PROTOCOL_SHA256,
    M03R_V12_SETTING_IDS,
    resolve_m03r_v12_setting,
)

M03R_V12_WORKER_PLAN_SCHEMA = "rl-quant.top2000-dev.m03r-v12-worker-plan-v1"
M03R_V12_PANEL_PLAN_SCHEMA = "rl-quant.top2000-dev.m03r-v12-panel-plan-v1"


class M03RV12PredictiveWorkerError(ValueError):
    """The v12 worker, panel, or paired-input identity drifted."""


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
        raise M03RV12PredictiveWorkerError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class M03RV12PredictiveWorkerPlan:
    setting_index: int
    setting_id: str
    output_root: str
    cache_path: str
    initial_parameter_state_path: str
    panel_episode_schedule_sha256: str
    initial_parameter_state_file_sha256: str
    initial_parameter_state_sha256: str
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
    seed: int = 17
    fold_count: int = 6
    expected_world_size: int = 2
    h100s_per_worker: int = 2
    predictive_optimizer_updates: int = 64
    economic_optimizer_updates: int = 0
    qualification_updates: tuple[int, ...] = (64,)
    selected_horizon_sessions: int = 3
    checkpoint_round_trip_required: bool = True
    paired_episode_schedule_required: bool = True
    paired_rank_shards_required: bool = True
    package_authorized: bool = False
    kubernetes_launch_authorized: bool = False
    outer_2026_access_authorized: bool = False
    protocol_generation: str = M03R_V12_PROTOCOL_GENERATION
    design_id: str = M03R_V12_DESIGN_ID
    protocol_sha256: str = M03R_V12_PROTOCOL_SHA256
    development_only: bool = True
    reportable: bool = False
    promotion_eligible: bool = False
    schema: str = M03R_V12_WORKER_PLAN_SCHEMA

    def validate(self) -> None:
        setting = resolve_m03r_v12_setting(self.setting_index)
        for name in (
            "panel_episode_schedule_sha256",
            "initial_parameter_state_file_sha256",
            "initial_parameter_state_sha256",
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
            or self.seed != M03R_V12_PREDICTIVE_SPEC.seed
            or self.fold_count != M03R_V12_PREDICTIVE_SPEC.chronological_fold_count
            or self.expected_world_size != M03R_V12_PREDICTIVE_SPEC.expected_world_size
            or self.h100s_per_worker != 2
            or self.predictive_optimizer_updates != 64
            or self.economic_optimizer_updates != 0
            or self.qualification_updates != (64,)
            or self.selected_horizon_sessions != 3
            or not self.checkpoint_round_trip_required
            or not self.paired_episode_schedule_required
            or not self.paired_rank_shards_required
            or self.package_authorized
            or self.kubernetes_launch_authorized
            or self.outer_2026_access_authorized
            or self.protocol_generation != M03R_V12_PROTOCOL_GENERATION
            or self.design_id != M03R_V12_DESIGN_ID
            or self.protocol_sha256 != M03R_V12_PROTOCOL_SHA256
            or not self.development_only
            or self.reportable
            or self.promotion_eligible
            or self.schema != M03R_V12_WORKER_PLAN_SCHEMA
        ):
            raise M03RV12PredictiveWorkerError("v12 worker plan drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class M03RV12PredictivePanelPlan:
    workers: tuple[M03RV12PredictiveWorkerPlan, ...]
    indexed_completions: int = 3
    parallelism: int = 3
    h100s_per_completion: int = 2
    maximum_h100_requests: int = 6
    economic_panel_authorized: bool = False
    package_authorized: bool = False
    kubernetes_launch_authorized: bool = False
    protocol_sha256: str = M03R_V12_PROTOCOL_SHA256
    schema: str = M03R_V12_PANEL_PLAN_SCHEMA

    def validate(self) -> None:
        for worker in self.workers:
            worker.validate()
        shared_fields = (
            "panel_episode_schedule_sha256",
            "initial_parameter_state_sha256",
            "initial_parameter_state_file_sha256",
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
            "selected_horizon_sessions",
        )
        if (
            len(self.workers) != 3
            or tuple(worker.setting_index for worker in self.workers) != (0, 1, 2)
            or tuple(worker.setting_id for worker in self.workers)
            != M03R_V12_SETTING_IDS
            or any(
                len({getattr(worker, field) for worker in self.workers}) != 1
                for field in shared_fields
            )
            or len({worker.output_root for worker in self.workers}) != 3
            or len({worker.receipt_sha256 for worker in self.workers}) != 3
            or self.indexed_completions != 3
            or self.parallelism != 3
            or self.h100s_per_completion != 2
            or self.maximum_h100_requests != 6
            or self.parallelism * self.h100s_per_completion
            != self.maximum_h100_requests
            or self.economic_panel_authorized
            or self.package_authorized
            or self.kubernetes_launch_authorized
            or self.protocol_sha256 != M03R_V12_PROTOCOL_SHA256
            or self.schema != M03R_V12_PANEL_PLAN_SCHEMA
        ):
            raise M03RV12PredictiveWorkerError("v12 panel plan drifted")

    @property
    def receipt_sha256(self) -> str:
        self.validate()
        return _sha256(asdict(self))


__all__ = [
    "M03R_V12_PANEL_PLAN_SCHEMA",
    "M03R_V12_WORKER_PLAN_SCHEMA",
    "M03RV12PredictivePanelPlan",
    "M03RV12PredictiveWorkerError",
    "M03RV12PredictiveWorkerPlan",
]
