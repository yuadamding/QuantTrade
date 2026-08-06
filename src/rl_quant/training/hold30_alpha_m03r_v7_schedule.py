"""Deterministic, fail-closed schedule renderer for the M03R v7 panel.

The renderer materializes fold/seed cells, setting workers, admission waves,
and immutable hashes.  It deliberately has no Kubernetes client, subprocess
entry point, or launch command.  Its output is a research scheduling receipt,
not authority to allocate a GPU.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, NoReturn

from rl_quant.protocol.hold30_alpha_m03r_v7_schedule import (
    M03R_V7_ADMISSION_ORDER,
    M03R_V7_AUXILIARY_CONTROLS,
    M03R_V7_DESIGN_ID,
    M03R_V7_H100_TOPOLOGY,
    M03R_V7_LAUNCH_BLOCKERS,
    M03R_V7_M00_PARITY_REFERENCE_SETTING_ID,
    M03R_V7_M01_PARITY_SETTING_ID,
    M03R_V7_PANEL,
    M03R_V7_PROTOCOL_GENERATION,
    M03R_V7_SCHEDULE_PROTOCOL_SHA256,
    M03R_V7_WAVE1_SETTING_INDICES,
    M03R_V7_WAVE2_SETTING_INDICES,
)

M03R_V7_EXPERIMENT_SCHEDULE_SCHEMA = "rl-quant.m03r-v7-experiment-schedule-v1"
M03R_V7_M01_PARITY_PLAN_SCHEMA = "rl-quant.m03r-v7-m01-parity-plan-v1"


class M03RV7ScheduleError(ValueError):
    """A rendered cell, worker, wave, or fail-closed receipt is inconsistent."""


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
        raise M03RV7ScheduleError(
            "v7 rendered schedule is not canonical-JSON safe"
        ) from exc
    return rendered.encode("utf-8")


def _sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class M03RV7FoldSeedCell:
    """One algorithmic replicate for one setting and one market fold."""

    global_cell_index: int
    setting_cell_index: int
    setting_index: int
    setting_id: str
    fold_index: int
    seed: int
    promotion_eligible: bool
    seed_is_independent_market_history: bool = False
    protocol_generation: str = M03R_V7_PROTOCOL_GENERATION
    design_id: str = M03R_V7_DESIGN_ID

    def __post_init__(self) -> None:
        if (
            self.protocol_generation != M03R_V7_PROTOCOL_GENERATION
            or self.design_id != M03R_V7_DESIGN_ID
        ):
            raise M03RV7ScheduleError("v7 fold/seed cell identity drifted")
        try:
            expected_setting_id = M03R_V7_PANEL.setting_ids[self.setting_index]
            seed_index = M03R_V7_PANEL.paired_seeds.index(self.seed)
        except (IndexError, ValueError) as exc:
            raise M03RV7ScheduleError(
                "v7 fold/seed cell uses an unknown setting index or seed"
            ) from exc
        if self.setting_id != expected_setting_id:
            raise M03RV7ScheduleError("v7 fold/seed cell setting ID drifted")
        if not 0 <= self.fold_index < M03R_V7_PANEL.fold_count:
            raise M03RV7ScheduleError("v7 fold index is out of range")
        expected_setting_cell = (
            self.fold_index * len(M03R_V7_PANEL.paired_seeds) + seed_index
        )
        expected_global_cell = (
            self.setting_index * M03R_V7_PANEL.cells_per_setting
            + expected_setting_cell
        )
        if (
            self.setting_cell_index != expected_setting_cell
            or self.global_cell_index != expected_global_cell
        ):
            raise M03RV7ScheduleError("v7 fold/seed cell ordering drifted")
        if self.promotion_eligible != M03R_V7_PANEL.promotion_eligible(
            self.setting_id
        ):
            raise M03RV7ScheduleError("v7 cell promotion eligibility drifted")
        if self.seed_is_independent_market_history:
            raise M03RV7ScheduleError(
                "v7 seeds are algorithmic replicates, not independent market paths"
            )

    @property
    def cell_id(self) -> str:
        return (
            f"setting-{self.setting_index:02d}/fold-{self.fold_index:02d}/"
            f"seed-{self.seed}"
        )


@dataclass(frozen=True, slots=True)
class M03RV7SettingWorkerPlan:
    """One two-H100 worker responsible for all thirty cells of one setting."""

    worker_index: int
    setting_index: int
    setting_id: str
    admission_wave: Literal[1, 2]
    initial_scheduler_state: Literal["admitted", "pending-backfill"]
    requested_h100_count: int
    torchrun_nproc_per_node: int
    complete_asset_cross_section_on_every_rank: bool
    distributed_batch_axis: Literal["origin-or-trajectory"]
    stock_axis_partitioning_allowed: bool
    cells: tuple[M03RV7FoldSeedCell, ...]
    production_entrypoint_available: bool
    launch_authorized: bool

    def __post_init__(self) -> None:
        try:
            expected_id = M03R_V7_PANEL.setting_ids[self.setting_index]
            expected_worker_index = M03R_V7_ADMISSION_ORDER.index(self.setting_index)
        except (IndexError, ValueError) as exc:
            raise M03RV7ScheduleError("v7 worker setting index is invalid") from exc
        if self.worker_index != expected_worker_index:
            raise M03RV7ScheduleError(
                "v7 worker index must equal the frozen admission rank"
            )
        if self.setting_id != expected_id:
            raise M03RV7ScheduleError("v7 worker setting identity drifted")
        if self.setting_index in M03R_V7_WAVE1_SETTING_INDICES:
            expected_wave = 1
            expected_state = "admitted"
        elif self.setting_index in M03R_V7_WAVE2_SETTING_INDICES:
            expected_wave = 2
            expected_state = "pending-backfill"
        else:  # pragma: no cover - frozen panel validation makes this unreachable
            raise M03RV7ScheduleError("v7 worker is absent from both waves")
        if (
            self.admission_wave != expected_wave
            or self.initial_scheduler_state != expected_state
        ):
            raise M03RV7ScheduleError("v7 worker admission priority drifted")
        if (
            self.requested_h100_count
            != M03R_V7_H100_TOPOLOGY.gpu_count_per_setting_worker
            or self.torchrun_nproc_per_node
            != M03R_V7_H100_TOPOLOGY.torchrun_nproc_per_node
            or not self.complete_asset_cross_section_on_every_rank
            or self.distributed_batch_axis != "origin-or-trajectory"
            or self.stock_axis_partitioning_allowed
        ):
            raise M03RV7ScheduleError("v7 two-H100 worker topology drifted")
        if len(self.cells) != M03R_V7_PANEL.cells_per_setting:
            raise M03RV7ScheduleError("each v7 worker must own exactly thirty cells")
        expected_cells = tuple(
            _build_cell(
                setting_index=self.setting_index,
                fold_index=fold_index,
                seed=seed,
            )
            for fold_index in range(M03R_V7_PANEL.fold_count)
            for seed in M03R_V7_PANEL.paired_seeds
        )
        if self.cells != expected_cells:
            raise M03RV7ScheduleError(
                "v7 worker cells must use the exact fold-major paired inventory"
            )
        if self.production_entrypoint_available or self.launch_authorized:
            raise M03RV7ScheduleError(
                "v7 rendered workers cannot claim an executable production launch"
            )

    @property
    def worker_id(self) -> str:
        return (
            f"m03r-v7-admission-{self.worker_index:02d}-"
            f"setting-{self.setting_index:02d}"
        )


def _build_cell(
    *, setting_index: int, fold_index: int, seed: int
) -> M03RV7FoldSeedCell:
    seed_index = M03R_V7_PANEL.paired_seeds.index(seed)
    setting_cell_index = fold_index * len(M03R_V7_PANEL.paired_seeds) + seed_index
    return M03RV7FoldSeedCell(
        global_cell_index=(
            setting_index * M03R_V7_PANEL.cells_per_setting + setting_cell_index
        ),
        setting_cell_index=setting_cell_index,
        setting_index=setting_index,
        setting_id=M03R_V7_PANEL.setting_ids[setting_index],
        fold_index=fold_index,
        seed=seed,
        promotion_eligible=M03R_V7_PANEL.promotion_eligible(
            M03R_V7_PANEL.setting_ids[setting_index]
        ),
    )


def _build_worker(setting_index: int) -> M03RV7SettingWorkerPlan:
    wave: Literal[1, 2]
    state: Literal["admitted", "pending-backfill"]
    if setting_index in M03R_V7_WAVE1_SETTING_INDICES:
        wave = 1
        state = "admitted"
    else:
        wave = 2
        state = "pending-backfill"
    return M03RV7SettingWorkerPlan(
        worker_index=M03R_V7_ADMISSION_ORDER.index(setting_index),
        setting_index=setting_index,
        setting_id=M03R_V7_PANEL.setting_ids[setting_index],
        admission_wave=wave,
        initial_scheduler_state=state,
        requested_h100_count=M03R_V7_H100_TOPOLOGY.gpu_count_per_setting_worker,
        torchrun_nproc_per_node=M03R_V7_H100_TOPOLOGY.torchrun_nproc_per_node,
        complete_asset_cross_section_on_every_rank=True,
        distributed_batch_axis="origin-or-trajectory",
        stock_axis_partitioning_allowed=False,
        cells=tuple(
            _build_cell(
                setting_index=setting_index,
                fold_index=fold_index,
                seed=seed,
            )
            for fold_index in range(M03R_V7_PANEL.fold_count)
            for seed in M03R_V7_PANEL.paired_seeds
        ),
        production_entrypoint_available=False,
        launch_authorized=False,
    )


@dataclass(frozen=True, slots=True)
class M03RV7ExperimentSchedule:
    """Content-bound primary-panel schedule with no mutation authority."""

    workers: tuple[M03RV7SettingWorkerPlan, ...]
    qualification_blockers: tuple[str, ...]
    schedule_protocol_sha256: str
    launch_authorized: bool
    cluster_mutation_authorized: bool
    schedule_sha256: str
    protocol_generation: str = M03R_V7_PROTOCOL_GENERATION
    design_id: str = M03R_V7_DESIGN_ID

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": M03R_V7_EXPERIMENT_SCHEDULE_SCHEMA,
            "protocol_generation": self.protocol_generation,
            "design_id": self.design_id,
            "schedule_protocol_sha256": self.schedule_protocol_sha256,
            "workers": [asdict(worker) for worker in self.workers],
            "qualification_blockers": list(self.qualification_blockers),
            "launch_authorized": self.launch_authorized,
            "cluster_mutation_authorized": self.cluster_mutation_authorized,
        }

    def __post_init__(self) -> None:
        if (
            self.protocol_generation != M03R_V7_PROTOCOL_GENERATION
            or self.design_id != M03R_V7_DESIGN_ID
            or self.schedule_protocol_sha256 != M03R_V7_SCHEDULE_PROTOCOL_SHA256
        ):
            raise M03RV7ScheduleError("v7 experiment schedule identity drifted")
        expected_workers = tuple(
            _build_worker(setting_index)
            for setting_index in M03R_V7_ADMISSION_ORDER
        )
        if self.workers != expected_workers:
            raise M03RV7ScheduleError(
                "v7 experiment workers were not canonically rendered"
            )
        if self.qualification_blockers != M03R_V7_LAUNCH_BLOCKERS:
            raise M03RV7ScheduleError("v7 schedule launch blockers drifted")
        if self.launch_authorized or self.cluster_mutation_authorized:
            raise M03RV7ScheduleError(
                "v7 schedule must remain launch-blocked and non-mutating"
            )
        if self.schedule_sha256 != _sha256(self.canonical_payload()):
            raise M03RV7ScheduleError("v7 experiment schedule hash mismatch")

    @property
    def cells(self) -> tuple[M03RV7FoldSeedCell, ...]:
        return tuple(
            sorted(
                (cell for worker in self.workers for cell in worker.cells),
                key=lambda cell: cell.global_cell_index,
            )
        )

    @property
    def initially_admitted_workers(self) -> tuple[M03RV7SettingWorkerPlan, ...]:
        return tuple(
            worker
            for worker in self.workers
            if worker.initial_scheduler_state == "admitted"
        )

    @property
    def pending_backfill_workers(self) -> tuple[M03RV7SettingWorkerPlan, ...]:
        return tuple(
            worker
            for worker in self.workers
            if worker.initial_scheduler_state == "pending-backfill"
        )


def build_m03r_v7_experiment_schedule() -> M03RV7ExperimentSchedule:
    """Render the immutable schedule without invoking any external system."""

    workers = tuple(
        _build_worker(setting_index)
        for setting_index in M03R_V7_ADMISSION_ORDER
    )
    fields: dict[str, Any] = {
        "workers": workers,
        "qualification_blockers": M03R_V7_LAUNCH_BLOCKERS,
        "schedule_protocol_sha256": M03R_V7_SCHEDULE_PROTOCOL_SHA256,
        "launch_authorized": False,
        "cluster_mutation_authorized": False,
        "protocol_generation": M03R_V7_PROTOCOL_GENERATION,
        "design_id": M03R_V7_DESIGN_ID,
    }
    unsigned = M03RV7ExperimentSchedule.__new__(M03RV7ExperimentSchedule)
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return M03RV7ExperimentSchedule(
        **fields,
        schedule_sha256=_sha256(unsigned.canonical_payload()),
    )


@dataclass(frozen=True, slots=True)
class M03RV7M01ParityPlan:
    """Short deterministic gradient-null qualification, never a full study."""

    optimizer_update_count: int
    seed: int
    requested_h100_count: int
    same_seed: bool
    same_minibatches: bool
    same_checkpoint_initialization: bool
    same_initial_optimizer_state: bool
    exact_gradient_parity_required: bool
    parameter_and_model_state_parity_required: bool
    optimizer_state_parity_required: bool
    complete_checkpoint_or_receipt_hash_equality_required: bool
    full_fold_seed_study: bool
    launch_authorized: bool
    plan_sha256: str
    reference_setting_id: str = M03R_V7_M00_PARITY_REFERENCE_SETTING_ID
    benchmark_subtraction_setting_id: str = M03R_V7_M01_PARITY_SETTING_ID
    protocol_generation: str = M03R_V7_PROTOCOL_GENERATION
    design_id: str = M03R_V7_DESIGN_ID

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": M03R_V7_M01_PARITY_PLAN_SCHEMA,
            "optimizer_update_count": self.optimizer_update_count,
            "seed": self.seed,
            "requested_h100_count": self.requested_h100_count,
            "same_seed": self.same_seed,
            "same_minibatches": self.same_minibatches,
            "same_checkpoint_initialization": self.same_checkpoint_initialization,
            "same_initial_optimizer_state": self.same_initial_optimizer_state,
            "exact_gradient_parity_required": self.exact_gradient_parity_required,
            "parameter_and_model_state_parity_required": (
                self.parameter_and_model_state_parity_required
            ),
            "optimizer_state_parity_required": self.optimizer_state_parity_required,
            "complete_checkpoint_or_receipt_hash_equality_required": (
                self.complete_checkpoint_or_receipt_hash_equality_required
            ),
            "full_fold_seed_study": self.full_fold_seed_study,
            "launch_authorized": self.launch_authorized,
            "reference_setting_id": self.reference_setting_id,
            "benchmark_subtraction_setting_id": (
                self.benchmark_subtraction_setting_id
            ),
            "protocol_generation": self.protocol_generation,
            "design_id": self.design_id,
        }

    def __post_init__(self) -> None:
        contract = M03R_V7_AUXILIARY_CONTROLS
        if (
            self.protocol_generation != M03R_V7_PROTOCOL_GENERATION
            or self.design_id != M03R_V7_DESIGN_ID
            or self.reference_setting_id != contract.m00_reference_setting_id
            or self.benchmark_subtraction_setting_id
            != contract.m01_gradient_null_setting_id
        ):
            raise M03RV7ScheduleError("v7 M01 parity plan identity drifted")
        if not (
            contract.m01_optimizer_update_minimum
            <= self.optimizer_update_count
            <= contract.m01_optimizer_update_maximum
        ):
            raise M03RV7ScheduleError("M01 parity requires exactly 2-4 updates")
        if (
            self.seed != M03R_V7_PANEL.paired_seeds[0]
            or self.requested_h100_count != contract.m01_gpu_count
            or not all(
                (
                    self.same_seed,
                    self.same_minibatches,
                    self.same_checkpoint_initialization,
                    self.same_initial_optimizer_state,
                    self.exact_gradient_parity_required,
                    self.parameter_and_model_state_parity_required,
                    self.optimizer_state_parity_required,
                )
            )
            or self.complete_checkpoint_or_receipt_hash_equality_required
            or self.full_fold_seed_study
            or self.launch_authorized
        ):
            raise M03RV7ScheduleError(
                "M01 parity must compare exact update state without requiring "
                "complete setting-specific receipt hashes"
            )
        if self.plan_sha256 != _sha256(self.canonical_payload()):
            raise M03RV7ScheduleError("v7 M01 parity plan hash mismatch")


def build_m03r_v7_m01_parity_plan(
    *, optimizer_update_count: int
) -> M03RV7M01ParityPlan:
    """Render a short M00/M01 parity plan; do not launch it."""

    fields: dict[str, Any] = {
        "optimizer_update_count": optimizer_update_count,
        "seed": M03R_V7_PANEL.paired_seeds[0],
        "requested_h100_count": M03R_V7_AUXILIARY_CONTROLS.m01_gpu_count,
        "same_seed": True,
        "same_minibatches": True,
        "same_checkpoint_initialization": True,
        "same_initial_optimizer_state": True,
        "exact_gradient_parity_required": True,
        "parameter_and_model_state_parity_required": True,
        "optimizer_state_parity_required": True,
        "complete_checkpoint_or_receipt_hash_equality_required": False,
        "full_fold_seed_study": False,
        "launch_authorized": False,
        "reference_setting_id": M03R_V7_M00_PARITY_REFERENCE_SETTING_ID,
        "benchmark_subtraction_setting_id": M03R_V7_M01_PARITY_SETTING_ID,
        "protocol_generation": M03R_V7_PROTOCOL_GENERATION,
        "design_id": M03R_V7_DESIGN_ID,
    }
    unsigned = M03RV7M01ParityPlan.__new__(M03RV7M01ParityPlan)
    for name, value in fields.items():
        object.__setattr__(unsigned, name, value)
    return M03RV7M01ParityPlan(
        **fields,
        plan_sha256=_sha256(unsigned.canonical_payload()),
    )


def require_m03r_v7_schedule_launch_ready() -> NoReturn:
    """Fail closed: this declarative renderer can never authorize a launch."""

    schedule = build_m03r_v7_experiment_schedule()
    raise M03RV7ScheduleError(
        "M03R v7 production/H100 launch remains blocked: "
        + ", ".join(schedule.qualification_blockers)
    )


__all__ = [
    "M03R_V7_EXPERIMENT_SCHEDULE_SCHEMA",
    "M03R_V7_M01_PARITY_PLAN_SCHEMA",
    "M03RV7ExperimentSchedule",
    "M03RV7FoldSeedCell",
    "M03RV7M01ParityPlan",
    "M03RV7ScheduleError",
    "M03RV7SettingWorkerPlan",
    "build_m03r_v7_experiment_schedule",
    "build_m03r_v7_m01_parity_plan",
    "require_m03r_v7_schedule_launch_ready",
]
