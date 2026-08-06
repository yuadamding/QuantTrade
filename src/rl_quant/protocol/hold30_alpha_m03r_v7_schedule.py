"""Immutable experiment and resource contract for the M03R v7 panel.

This module describes research intent only.  It cannot submit Kubernetes jobs,
read a lockbox, or turn missing production qualifications into launch authority.
The primary panel contains twelve paired mechanism settings.  Two additional
governance controls remain outside that panel: M01 is a short gradient-parity
qualification and A05 is reserve-only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

from rl_quant.protocol.hold30_alpha_m03r_v7 import (
    M03R_V7_A05_RESERVE_SETTING_ID,
    M03R_V7_CANONICAL_SETTING_ID,
    M03R_V7_DESIGN_ID,
    M03R_V7_FIXED_TE_FLOOR_RESERVE,
    M03R_V7_GRADIENT_NULL_QUALIFICATION,
    M03R_V7_M00_QUALIFICATION_SETTING_ID,
    M03R_V7_M01_QUALIFICATION_SETTING_ID,
    M03R_V7_PRIMARY_SETTING_IDS,
    M03R_V7_PROTOCOL_GENERATION,
    M03R_V7_SHARED_CONFIGURATION,
)

M03R_V7_WAVE1_SETTING_INDICES = (0, 1, 2, 3, 4, 5, 6, 8)
M03R_V7_WAVE2_SETTING_INDICES = (7, 9, 10, 11)
M03R_V7_ADMISSION_ORDER = (
    M03R_V7_WAVE1_SETTING_INDICES + M03R_V7_WAVE2_SETTING_INDICES
)
M03R_V7_PAIRED_SEEDS = M03R_V7_SHARED_CONFIGURATION.paired_seeds

M03R_V7_M00_PARITY_REFERENCE_SETTING_ID = M03R_V7_M00_QUALIFICATION_SETTING_ID
M03R_V7_M01_PARITY_SETTING_ID = M03R_V7_M01_QUALIFICATION_SETTING_ID

M03R_V7_LAUNCH_BLOCKERS = (
    "governed-v7-twelve-setting-production-driver-not-implemented",
    "point-in-time-active300-data-and-benchmark-manifests-not-sealed",
    "two-stage-confidence-calibration-not-receipted",
    "five-seed-output-space-ensemble-not-receipted",
    "cause-typed-chronological-ledger-adapter-not-qualified",
    "risk-projection-and-empirical-execution-receipts-not-qualified",
    "inference-and-multiplicity-family-not-sealed",
    "cuda-two-rank-ddp-restart-parity-not-qualified",
    "h100-80gb-capacity-and-throughput-not-qualified",
)

M03R_V7_SCHEDULE_PROTOCOL_SCHEMA = "rl-quant.m03r-v7-schedule-protocol-v1"


class M03RV7ScheduleProtocolError(ValueError):
    """The frozen v7 scheduling or resource geometry is inconsistent."""


def _canonical_json(payload: dict[str, Any]) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise M03RV7ScheduleProtocolError(
            "v7 schedule payload is not canonical-JSON safe"
        ) from exc
    return rendered.encode("utf-8")


def _sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV7PanelContract:
    """Paired scientific inventory for the primary twelve-setting panel."""

    setting_ids: tuple[str, ...] = M03R_V7_PRIMARY_SETTING_IDS
    fold_count: int = M03R_V7_SHARED_CONFIGURATION.validation_fold_count
    paired_seeds: tuple[int, ...] = M03R_V7_PAIRED_SEEDS
    wave1_setting_indices: tuple[int, ...] = M03R_V7_WAVE1_SETTING_INDICES
    wave2_setting_indices: tuple[int, ...] = M03R_V7_WAVE2_SETTING_INDICES
    canonical_setting_id: str = M03R_V7_CANONICAL_SETTING_ID
    only_canonical_is_promotion_eligible: bool = True
    seed_cells_are_independent_market_histories: bool = False
    one_deployed_ensemble_return_path_per_fold: bool = True
    checkpoint_selection_occurs_independently_per_seed: bool = True
    ensemble_member_count: int = 5

    def __post_init__(self) -> None:
        if self.setting_ids != M03R_V7_PRIMARY_SETTING_IDS:
            raise M03RV7ScheduleProtocolError(
                "v7 primary setting order must remain immutable"
            )
        if len(self.setting_ids) != 12 or len(set(self.setting_ids)) != 12:
            raise M03RV7ScheduleProtocolError(
                "v7 primary panel must contain twelve unique settings"
            )
        if self.fold_count != 6 or self.paired_seeds != M03R_V7_PAIRED_SEEDS:
            raise M03RV7ScheduleProtocolError(
                "v7 requires six folds and the five frozen paired seeds"
            )
        if (
            self.wave1_setting_indices != M03R_V7_WAVE1_SETTING_INDICES
            or self.wave2_setting_indices != M03R_V7_WAVE2_SETTING_INDICES
        ):
            raise M03RV7ScheduleProtocolError("v7 admission waves drifted")
        wave1 = set(self.wave1_setting_indices)
        wave2 = set(self.wave2_setting_indices)
        if wave1 & wave2 or wave1 | wave2 != set(range(len(self.setting_ids))):
            raise M03RV7ScheduleProtocolError(
                "v7 waves must be disjoint and cover all twelve settings"
            )
        if (
            self.canonical_setting_id != M03R_V7_CANONICAL_SETTING_ID
            or not self.only_canonical_is_promotion_eligible
        ):
            raise M03RV7ScheduleProtocolError(
                "only the canonical v7 setting may be promotion eligible"
            )
        if (
            self.seed_cells_are_independent_market_histories
            or not self.one_deployed_ensemble_return_path_per_fold
            or not self.checkpoint_selection_occurs_independently_per_seed
            or self.ensemble_member_count != len(self.paired_seeds)
        ):
            raise M03RV7ScheduleProtocolError(
                "v7 seeds are paired algorithmic replicates, followed by one "
                "five-seed deployed ensemble path per fold"
            )

    @property
    def cells_per_setting(self) -> int:
        return self.fold_count * len(self.paired_seeds)

    @property
    def total_fold_seed_cells(self) -> int:
        return len(self.setting_ids) * self.cells_per_setting

    def promotion_eligible(self, setting_id: str) -> bool:
        if setting_id not in self.setting_ids:
            raise M03RV7ScheduleProtocolError(
                f"unknown v7 primary setting {setting_id!r}"
            )
        return setting_id == self.canonical_setting_id


@dataclass(frozen=True, slots=True)
class M03RV7H100TopologyContract:
    """Two-rank, full-cross-section topology under the sixteen-H100 ceiling."""

    gpu_product: str = "NVIDIA-H100-80GB-HBM3"
    gpu_memory_gib_per_device: int = 80
    gpu_count_per_setting_worker: int = 2
    h100_cluster_cap: int = 16
    maximum_concurrent_setting_workers: int = 8
    primary_setting_worker_count: int = 12
    torchrun_nproc_per_node: int = 2
    distributed_mode: Literal["ddp-synchronized-trajectory-training"] = (
        "ddp-synchronized-trajectory-training"
    )
    complete_asset_cross_section_on_every_rank: bool = True
    distributed_batch_axis: Literal["origin-or-trajectory"] = "origin-or-trajectory"
    stock_axis_partitioning_allowed: bool = False
    gradient_all_reduce_frequency: Literal["once-per-effective-batch"] = (
        "once-per-effective-batch"
    )
    activation_checkpointing_required: bool = True
    raw_stock_chunking_required: bool = True
    model_parameter_sharding_required: bool = False
    wave2_admission: Literal["scheduler-pending-automatic-backfill"] = (
        "scheduler-pending-automatic-backfill"
    )

    def __post_init__(self) -> None:
        expected = {
            "gpu_product": "NVIDIA-H100-80GB-HBM3",
            "gpu_memory_gib_per_device": 80,
            "gpu_count_per_setting_worker": 2,
            "h100_cluster_cap": 16,
            "maximum_concurrent_setting_workers": 8,
            "primary_setting_worker_count": 12,
            "torchrun_nproc_per_node": 2,
            "distributed_mode": "ddp-synchronized-trajectory-training",
            "complete_asset_cross_section_on_every_rank": True,
            "distributed_batch_axis": "origin-or-trajectory",
            "stock_axis_partitioning_allowed": False,
            "gradient_all_reduce_frequency": "once-per-effective-batch",
            "activation_checkpointing_required": True,
            "raw_stock_chunking_required": True,
            "model_parameter_sharding_required": False,
            "wave2_admission": "scheduler-pending-automatic-backfill",
        }
        for name, required in expected.items():
            if getattr(self, name) != required:
                raise M03RV7ScheduleProtocolError(
                    f"v7 H100 topology field {name} must be {required!r}"
                )
        if (
            self.maximum_concurrent_setting_workers
            * self.gpu_count_per_setting_worker
            != self.h100_cluster_cap
        ):
            raise M03RV7ScheduleProtocolError(
                "v7 concurrent workers must consume exactly the sixteen-H100 cap"
            )
        if self.primary_setting_worker_count <= self.maximum_concurrent_setting_workers:
            raise M03RV7ScheduleProtocolError(
                "v7 must preserve four initially pending backfill workers"
            )

    @property
    def requested_h100_count_for_all_settings(self) -> int:
        return self.primary_setting_worker_count * self.gpu_count_per_setting_worker

    @property
    def initial_running_h100_count(self) -> int:
        return (
            self.maximum_concurrent_setting_workers
            * self.gpu_count_per_setting_worker
        )

    @property
    def initially_pending_setting_worker_count(self) -> int:
        return self.primary_setting_worker_count - self.maximum_concurrent_setting_workers


@dataclass(frozen=True, slots=True)
class M03RV7AuxiliaryControlContract:
    """Controls deliberately excluded from the full 360-cell H100 panel."""

    m00_reference_setting_id: str = M03R_V7_M00_PARITY_REFERENCE_SETTING_ID
    m01_gradient_null_setting_id: str = M03R_V7_M01_PARITY_SETTING_ID
    m01_optimizer_update_minimum: int = (
        M03R_V7_GRADIENT_NULL_QUALIFICATION.minimum_optimizer_updates
    )
    m01_optimizer_update_maximum: int = (
        M03R_V7_GRADIENT_NULL_QUALIFICATION.maximum_optimizer_updates
    )
    m01_gpu_count: int = 2
    m01_requires_same_seed: bool = True
    m01_requires_same_minibatches: bool = True
    m01_requires_same_checkpoint_initialization: bool = True
    m01_requires_same_initial_optimizer_state: bool = True
    m01_requires_exact_gradient_parity: bool = True
    m01_requires_parameter_and_model_state_parity: bool = True
    m01_requires_optimizer_state_parity: bool = True
    m01_requires_complete_checkpoint_or_receipt_hash_equality: bool = False
    m01_full_fold_seed_study_permitted: bool = False
    a05_reserve_setting_id: str = M03R_V7_A05_RESERVE_SETTING_ID
    a05_primary_panel_member: bool = False
    a05_promotion_eligible: bool = False
    a05_automatic_launch_permitted: bool = False
    a05_trigger: str = M03R_V7_FIXED_TE_FLOOR_RESERVE.activation_condition

    def __post_init__(self) -> None:
        if (
            self.m00_reference_setting_id
            != M03R_V7_M00_PARITY_REFERENCE_SETTING_ID
            or self.m01_gradient_null_setting_id != M03R_V7_M01_PARITY_SETTING_ID
            or self.m01_optimizer_update_minimum != 2
            or self.m01_optimizer_update_maximum != 4
            or self.m01_gpu_count != 2
        ):
            raise M03RV7ScheduleProtocolError("v7 M01 parity geometry drifted")
        if not all(
            (
                self.m01_requires_same_seed,
                self.m01_requires_same_minibatches,
                self.m01_requires_same_checkpoint_initialization,
                self.m01_requires_same_initial_optimizer_state,
                self.m01_requires_exact_gradient_parity,
                self.m01_requires_parameter_and_model_state_parity,
                self.m01_requires_optimizer_state_parity,
            )
        ) or (
            self.m01_requires_complete_checkpoint_or_receipt_hash_equality
            or self.m01_full_fold_seed_study_permitted
        ):
            raise M03RV7ScheduleProtocolError(
                "M01 must remain a short deterministic gradient-null qualification"
            )
        core_parity = M03R_V7_GRADIENT_NULL_QUALIFICATION
        if (
            self.m01_requires_same_seed != core_parity.same_seed_required
            or self.m01_requires_same_minibatches
            != core_parity.same_minibatches_required
            or self.m01_requires_same_checkpoint_initialization
            != core_parity.same_checkpoint_initialization_required
            or self.m01_requires_same_initial_optimizer_state
            != core_parity.same_optimizer_state_required
            or self.m01_requires_exact_gradient_parity
            != core_parity.compare_gradients
            or self.m01_requires_parameter_and_model_state_parity
            != (
                core_parity.compare_parameter_updates
                and core_parity.compare_model_state_hashes
            )
            or self.m01_requires_optimizer_state_parity
            != core_parity.compare_optimizer_state_hashes
            or self.m01_requires_complete_checkpoint_or_receipt_hash_equality
            != core_parity.complete_checkpoint_or_receipt_hash_equality_required
        ):
            raise M03RV7ScheduleProtocolError(
                "v7 schedule parity semantics differ from the core protocol"
            )
        if (
            self.a05_reserve_setting_id != M03R_V7_A05_RESERVE_SETTING_ID
            or self.a05_primary_panel_member
            or self.a05_promotion_eligible
            or self.a05_automatic_launch_permitted
            or self.a05_trigger != M03R_V7_FIXED_TE_FLOOR_RESERVE.activation_condition
        ):
            raise M03RV7ScheduleProtocolError(
                "A05 must remain nonpromotable, reserve-only, and never automatic"
            )
        excluded = {
            self.m00_reference_setting_id,
            self.m01_gradient_null_setting_id,
            self.a05_reserve_setting_id,
        }
        if excluded & set(M03R_V7_PRIMARY_SETTING_IDS):
            raise M03RV7ScheduleProtocolError(
                "M00/M01/A05 auxiliary controls cannot enter the primary panel"
            )


@dataclass(frozen=True, slots=True)
class M03RV7LaunchGateContract:
    """Fail-closed status until every declared production prerequisite exists."""

    blockers: tuple[str, ...] = M03R_V7_LAUNCH_BLOCKERS
    launch_authorized: bool = False
    cluster_mutation_authorized: bool = False

    def __post_init__(self) -> None:
        if self.blockers != M03R_V7_LAUNCH_BLOCKERS or len(set(self.blockers)) != len(
            self.blockers
        ):
            raise M03RV7ScheduleProtocolError("v7 launch blockers drifted")
        if self.launch_authorized or self.cluster_mutation_authorized:
            raise M03RV7ScheduleProtocolError(
                "v7 schedule is declarative and must remain launch-blocked"
            )


M03R_V7_PANEL = M03RV7PanelContract()
M03R_V7_H100_TOPOLOGY = M03RV7H100TopologyContract()
M03R_V7_AUXILIARY_CONTROLS = M03RV7AuxiliaryControlContract()
M03R_V7_LAUNCH_GATE = M03RV7LaunchGateContract()


def m03r_v7_schedule_protocol_payload() -> dict[str, Any]:
    """Return the exact JSON-ready scheduling contract; perform no launch action."""

    return {
        "schema": M03R_V7_SCHEDULE_PROTOCOL_SCHEMA,
        "protocol_generation": M03R_V7_PROTOCOL_GENERATION,
        "design_id": M03R_V7_DESIGN_ID,
        "panel": asdict(M03R_V7_PANEL),
        "topology": asdict(M03R_V7_H100_TOPOLOGY),
        "auxiliary_controls": asdict(M03R_V7_AUXILIARY_CONTROLS),
        "launch_gate": asdict(M03R_V7_LAUNCH_GATE),
    }


M03R_V7_SCHEDULE_PROTOCOL_SHA256 = _sha256(m03r_v7_schedule_protocol_payload())


__all__ = [
    "M03R_V7_A05_RESERVE_SETTING_ID",
    "M03R_V7_ADMISSION_ORDER",
    "M03R_V7_AUXILIARY_CONTROLS",
    "M03R_V7_CANONICAL_SETTING_ID",
    "M03R_V7_DESIGN_ID",
    "M03R_V7_H100_TOPOLOGY",
    "M03R_V7_LAUNCH_BLOCKERS",
    "M03R_V7_LAUNCH_GATE",
    "M03R_V7_M00_PARITY_REFERENCE_SETTING_ID",
    "M03R_V7_M01_PARITY_SETTING_ID",
    "M03R_V7_PAIRED_SEEDS",
    "M03R_V7_PANEL",
    "M03R_V7_PRIMARY_SETTING_IDS",
    "M03R_V7_PROTOCOL_GENERATION",
    "M03R_V7_SCHEDULE_PROTOCOL_SCHEMA",
    "M03R_V7_SCHEDULE_PROTOCOL_SHA256",
    "M03R_V7_WAVE1_SETTING_INDICES",
    "M03R_V7_WAVE2_SETTING_INDICES",
    "M03RV7AuxiliaryControlContract",
    "M03RV7H100TopologyContract",
    "M03RV7LaunchGateContract",
    "M03RV7PanelContract",
    "M03RV7ScheduleProtocolError",
    "m03r_v7_schedule_protocol_payload",
]
