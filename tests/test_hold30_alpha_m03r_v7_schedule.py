"""Deterministic tests for the fail-closed M03R v7 H100 schedule."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v7 import (
    M03R_V7_A05_RESERVE_SETTING_ID as CORE_A05_ID,
)
from rl_quant.protocol.hold30_alpha_m03r_v7 import (
    M03R_V7_CANONICAL_SETTING_ID as CORE_CANONICAL_ID,
)
from rl_quant.protocol.hold30_alpha_m03r_v7 import (
    M03R_V7_DESIGN_ID as CORE_DESIGN_ID,
)
from rl_quant.protocol.hold30_alpha_m03r_v7 import (
    M03R_V7_GRADIENT_NULL_QUALIFICATION,
)
from rl_quant.protocol.hold30_alpha_m03r_v7 import (
    M03R_V7_M00_QUALIFICATION_SETTING_ID as CORE_M00_ID,
)
from rl_quant.protocol.hold30_alpha_m03r_v7 import (
    M03R_V7_M01_QUALIFICATION_SETTING_ID as CORE_M01_ID,
)
from rl_quant.protocol.hold30_alpha_m03r_v7 import (
    M03R_V7_PRIMARY_SETTING_IDS as CORE_PRIMARY_SETTING_IDS,
)
from rl_quant.protocol.hold30_alpha_m03r_v7 import (
    M03R_V7_PROTOCOL_GENERATION as CORE_PROTOCOL_GENERATION,
)
from rl_quant.protocol.hold30_alpha_m03r_v7_schedule import (
    M03R_V7_A05_RESERVE_SETTING_ID,
    M03R_V7_ADMISSION_ORDER,
    M03R_V7_AUXILIARY_CONTROLS,
    M03R_V7_CANONICAL_SETTING_ID,
    M03R_V7_DESIGN_ID,
    M03R_V7_H100_TOPOLOGY,
    M03R_V7_LAUNCH_BLOCKERS,
    M03R_V7_LAUNCH_GATE,
    M03R_V7_M00_PARITY_REFERENCE_SETTING_ID,
    M03R_V7_M01_PARITY_SETTING_ID,
    M03R_V7_PANEL,
    M03R_V7_PRIMARY_SETTING_IDS,
    M03R_V7_PROTOCOL_GENERATION,
    M03R_V7_SCHEDULE_PROTOCOL_SHA256,
    M03R_V7_WAVE1_SETTING_INDICES,
    M03R_V7_WAVE2_SETTING_INDICES,
    M03RV7H100TopologyContract,
    M03RV7PanelContract,
    M03RV7ScheduleProtocolError,
    m03r_v7_schedule_protocol_payload,
)
from rl_quant.training.hold30_alpha_m03r_v7_schedule import (
    M03RV7ScheduleError,
    build_m03r_v7_experiment_schedule,
    build_m03r_v7_m01_parity_plan,
    require_m03r_v7_schedule_launch_ready,
)

EXPECTED_SETTINGS = (
    "M03R-soft-persistence-active-alpha-hold30-v7",
    "P00-no-soft-persistence-v7",
    "P10-soft-persistence-10bp-v7",
    "A08-fixed-exit-hazard-v7",
    "A11-no-exact-hold-atom-v7",
    "A09-no-long-context-v7",
    "M02-active-risk-no-alpha-heads-v7",
    "A04-no-downside-score-adjustment-v7",
    "A12-fixed-2pct-active-risk-budget-v7",
    "A10-no-factor-neutral-projection-v7",
    "A06-sharpe-overlay-v7",
    "A07-direct-sharpe-v7",
)


def test_schedule_identity_is_sourced_exactly_from_core_v7_protocol() -> None:
    assert M03R_V7_PROTOCOL_GENERATION == CORE_PROTOCOL_GENERATION
    assert M03R_V7_DESIGN_ID == CORE_DESIGN_ID
    assert M03R_V7_CANONICAL_SETTING_ID == CORE_CANONICAL_ID
    assert M03R_V7_PRIMARY_SETTING_IDS == CORE_PRIMARY_SETTING_IDS
    assert M03R_V7_PRIMARY_SETTING_IDS == EXPECTED_SETTINGS
    assert M03R_V7_M00_PARITY_REFERENCE_SETTING_ID == CORE_M00_ID
    assert M03R_V7_M01_PARITY_SETTING_ID == CORE_M01_ID
    assert M03R_V7_A05_RESERVE_SETTING_ID == CORE_A05_ID


def test_primary_panel_freezes_six_folds_five_paired_seeds_and_360_cells() -> None:
    assert M03R_V7_PANEL.fold_count == 6
    assert M03R_V7_PANEL.paired_seeds == (17, 29, 43, 71, 101)
    assert M03R_V7_PANEL.cells_per_setting == 30
    assert M03R_V7_PANEL.total_fold_seed_cells == 360
    assert not M03R_V7_PANEL.seed_cells_are_independent_market_histories
    assert M03R_V7_PANEL.one_deployed_ensemble_return_path_per_fold
    assert M03R_V7_PANEL.checkpoint_selection_occurs_independently_per_seed
    assert M03R_V7_PANEL.ensemble_member_count == 5


def test_two_h100_worker_topology_uses_full_cross_section_and_batch_split() -> None:
    topology = M03R_V7_H100_TOPOLOGY
    assert topology.gpu_product == "NVIDIA-H100-80GB-HBM3"
    assert topology.gpu_memory_gib_per_device == 80
    assert topology.gpu_count_per_setting_worker == 2
    assert topology.torchrun_nproc_per_node == 2
    assert topology.distributed_mode == "ddp-synchronized-trajectory-training"
    assert topology.complete_asset_cross_section_on_every_rank
    assert topology.distributed_batch_axis == "origin-or-trajectory"
    assert not topology.stock_axis_partitioning_allowed
    assert topology.gradient_all_reduce_frequency == "once-per-effective-batch"
    assert topology.activation_checkpointing_required
    assert topology.raw_stock_chunking_required
    assert not topology.model_parameter_sharding_required


def test_sixteen_h100_cap_admits_wave1_and_leaves_wave2_pending_for_backfill() -> None:
    assert M03R_V7_WAVE1_SETTING_INDICES == (0, 1, 2, 3, 4, 5, 6, 8)
    assert M03R_V7_WAVE2_SETTING_INDICES == (7, 9, 10, 11)
    topology = M03R_V7_H100_TOPOLOGY
    assert topology.h100_cluster_cap == 16
    assert topology.maximum_concurrent_setting_workers == 8
    assert topology.requested_h100_count_for_all_settings == 24
    assert topology.initial_running_h100_count == 16
    assert topology.initially_pending_setting_worker_count == 4
    assert topology.wave2_admission == "scheduler-pending-automatic-backfill"

    schedule = build_m03r_v7_experiment_schedule()
    assert M03R_V7_ADMISSION_ORDER == (0, 1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11)
    assert tuple(worker.worker_index for worker in schedule.workers) == tuple(range(12))
    assert tuple(worker.setting_index for worker in schedule.workers) == (
        M03R_V7_ADMISSION_ORDER
    )
    assert tuple(worker.setting_index for worker in schedule.initially_admitted_workers) == (
        M03R_V7_WAVE1_SETTING_INDICES
    )
    assert tuple(worker.setting_index for worker in schedule.pending_backfill_workers) == (
        M03R_V7_WAVE2_SETTING_INDICES
    )
    assert sum(
        worker.requested_h100_count for worker in schedule.initially_admitted_workers
    ) == 16
    assert sum(
        worker.requested_h100_count for worker in schedule.pending_backfill_workers
    ) == 8


def test_fold_seed_cells_are_fold_major_paired_and_globally_unique() -> None:
    schedule = build_m03r_v7_experiment_schedule()
    assert len(schedule.workers) == 12
    assert len(schedule.cells) == 360
    assert tuple(cell.global_cell_index for cell in schedule.cells) == tuple(range(360))
    assert len({cell.cell_id for cell in schedule.cells}) == 360

    for worker in schedule.workers:
        assert len(worker.cells) == 30
        assert tuple((cell.fold_index, cell.seed) for cell in worker.cells) == tuple(
            (fold_index, seed)
            for fold_index in range(6)
            for seed in (17, 29, 43, 71, 101)
        )
        assert all(cell.setting_id == worker.setting_id for cell in worker.cells)
        assert all(not cell.seed_is_independent_market_history for cell in worker.cells)


def test_only_canonical_cells_are_promotion_eligible() -> None:
    schedule = build_m03r_v7_experiment_schedule()
    eligible_settings = {
        cell.setting_id for cell in schedule.cells if cell.promotion_eligible
    }
    assert eligible_settings == {M03R_V7_CANONICAL_SETTING_ID}
    assert sum(cell.promotion_eligible for cell in schedule.cells) == 30
    for setting_id in M03R_V7_PRIMARY_SETTING_IDS:
        assert M03R_V7_PANEL.promotion_eligible(setting_id) == (
            setting_id == M03R_V7_CANONICAL_SETTING_ID
        )
    with pytest.raises(M03RV7ScheduleProtocolError, match="unknown"):
        M03R_V7_PANEL.promotion_eligible(M03R_V7_M01_PARITY_SETTING_ID)


def test_m01_is_short_parity_only_and_complete_receipt_hashes_may_differ() -> None:
    for update_count in (2, 3, 4):
        plan = build_m03r_v7_m01_parity_plan(
            optimizer_update_count=update_count
        )
        assert plan.requested_h100_count == 2
        assert plan.seed == 17
        assert plan.same_seed and plan.same_minibatches
        assert plan.same_checkpoint_initialization
        assert plan.same_initial_optimizer_state
        assert plan.exact_gradient_parity_required
        assert plan.parameter_and_model_state_parity_required
        assert plan.optimizer_state_parity_required
        assert not plan.complete_checkpoint_or_receipt_hash_equality_required
        assert not plan.full_fold_seed_study
        assert not plan.launch_authorized
        assert plan.plan_sha256

    for invalid_count in (1, 5, 30):
        with pytest.raises(M03RV7ScheduleError, match="2-4 updates"):
            build_m03r_v7_m01_parity_plan(
                optimizer_update_count=invalid_count
            )
    assert not (
        M03R_V7_GRADIENT_NULL_QUALIFICATION
        .complete_checkpoint_or_receipt_hash_equality_required
    )


def test_m01_and_a05_never_enter_primary_or_automatic_schedule() -> None:
    excluded = {
        M03R_V7_M00_PARITY_REFERENCE_SETTING_ID,
        M03R_V7_M01_PARITY_SETTING_ID,
        M03R_V7_A05_RESERVE_SETTING_ID,
    }
    assert not excluded & set(M03R_V7_PRIMARY_SETTING_IDS)
    assert not M03R_V7_AUXILIARY_CONTROLS.m01_full_fold_seed_study_permitted
    assert not M03R_V7_AUXILIARY_CONTROLS.a05_primary_panel_member
    assert not M03R_V7_AUXILIARY_CONTROLS.a05_promotion_eligible
    assert not M03R_V7_AUXILIARY_CONTROLS.a05_automatic_launch_permitted
    assert "near-zero-active-risk" in M03R_V7_AUXILIARY_CONTROLS.a05_trigger

    scheduled_ids = {
        worker.setting_id for worker in build_m03r_v7_experiment_schedule().workers
    }
    assert not excluded & scheduled_ids


def test_schedule_is_deterministic_content_bound_and_rejects_mutation() -> None:
    first = build_m03r_v7_experiment_schedule()
    second = build_m03r_v7_experiment_schedule()
    assert first == second
    assert first.schedule_sha256 == second.schedule_sha256
    assert first.schedule_protocol_sha256 == M03R_V7_SCHEDULE_PROTOCOL_SHA256
    assert m03r_v7_schedule_protocol_payload() == (
        m03r_v7_schedule_protocol_payload()
    )

    with pytest.raises(M03RV7ScheduleError, match="canonically rendered"):
        replace(first, workers=tuple(reversed(first.workers)))
    with pytest.raises(M03RV7ScheduleError, match="hash mismatch"):
        replace(first, schedule_sha256="0" * 64)
    with pytest.raises(M03RV7ScheduleError, match="topology"):
        replace(first.workers[0], requested_h100_count=4)
    with pytest.raises(M03RV7ScheduleProtocolError, match="h100_cluster_cap"):
        replace(M03R_V7_H100_TOPOLOGY, h100_cluster_cap=24)
    with pytest.raises(M03RV7ScheduleProtocolError, match="setting order"):
        replace(
            M03R_V7_PANEL,
            setting_ids=tuple(reversed(M03R_V7_PANEL.setting_ids)),
        )


def test_schedule_is_purely_declarative_and_fails_closed() -> None:
    schedule = build_m03r_v7_experiment_schedule()
    assert schedule.qualification_blockers == M03R_V7_LAUNCH_BLOCKERS
    assert M03R_V7_LAUNCH_GATE.blockers == M03R_V7_LAUNCH_BLOCKERS
    assert not schedule.launch_authorized
    assert not schedule.cluster_mutation_authorized
    assert not M03R_V7_LAUNCH_GATE.launch_authorized
    assert not M03R_V7_LAUNCH_GATE.cluster_mutation_authorized
    assert all(not worker.production_entrypoint_available for worker in schedule.workers)
    assert all(not worker.launch_authorized for worker in schedule.workers)

    with pytest.raises(M03RV7ScheduleError, match="launch remains blocked"):
        require_m03r_v7_schedule_launch_ready()
    with pytest.raises(M03RV7ScheduleError, match="non-mutating"):
        replace(schedule, cluster_mutation_authorized=True)


def test_contract_rejects_oversubscription_and_seed_relabeling() -> None:
    with pytest.raises(
        M03RV7ScheduleProtocolError,
        match="maximum_concurrent_setting_workers",
    ):
        M03RV7H100TopologyContract(maximum_concurrent_setting_workers=9)
    with pytest.raises(M03RV7ScheduleProtocolError, match="five frozen paired seeds"):
        M03RV7PanelContract(paired_seeds=(17, 29, 43, 71, 999))
