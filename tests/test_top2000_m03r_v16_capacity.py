from __future__ import annotations

from dataclasses import replace
import inspect

import pytest

from rl_quant.training.top2000_m03r_v16_capacity import (
    M03RV16CapacityRankEvidence,
    build_m03r_v16_capacity_terminal,
)
from rl_quant.training import top2000_m03r_v16_capacity as capacity


def _rank(rank: int) -> M03RV16CapacityRankEvidence:
    return M03RV16CapacityRankEvidence(
        setting_index=2,
        distributed_rank=rank,
        distributed_world_size=2,
        cuda_device_name="NVIDIA H100 80GB HBM3",
        cuda_total_memory_bytes=80 * 1024**3,
        peak_allocated_bytes=40 * 1024**3 + rank,
        peak_reserved_bytes=50 * 1024**3 + rank,
        update_plan_sha256="1" * 64,
        batch_receipt_sha256=("2" if rank == 0 else "3") * 64,
        score_step_receipt_sha256=("4" if rank == 0 else "5") * 64,
        structural_slab_receipt_sha256="6" * 64,
        qualification_projection_receipt_sha256=("7" if rank == 0 else "8") * 64,
        qualification_requested_active_one_way_mass=0.01,
        qualification_projected_active_one_way_mass=0.0025,
        qualification_requested_to_executed_retention=0.25,
        post_update_model_state_sha256="9" * 64,
        post_update_optimizer_state_sha256="a" * 64,
        episode_state_rows=345,
        global_origin_count=43,
        local_origin_count=22 if rank == 0 else 21,
    )


def test_v16_capacity_terminal_requires_exact_update_projection_and_rank_equality() -> (
    None
):
    terminal = build_m03r_v16_capacity_terminal((_rank(1), _rank(0)))
    terminal.validate()
    assert tuple(value.distributed_rank for value in terminal.rank_evidence) == (0, 1)
    assert terminal.exact_workload_qualified
    assert not terminal.predictive_training_authorized
    assert not terminal.scientific_checkpoint_published

    with pytest.raises(Exception, match="terminal drifted"):
        build_m03r_v16_capacity_terminal(
            (
                _rank(0),
                replace(_rank(1), post_update_model_state_sha256="b" * 64),
            )
        )


def test_v16_capacity_runner_owns_the_real_projection_call() -> None:
    source = inspect.getsource(capacity.run_m03r_v16_disposable_capacity_rank)
    assert "run_m03r_v16_pretraining_fold_update" in source
    assert "project_m03r_v9_active_book" in source
    assert "qualification_projection_probe" not in source


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("bf16_forward_backward_executed", False),
        ("nccl_gradient_sum_executed", False),
        ("optimizer_mutation_executed", False),
        ("qualification_projection_executed", False),
        ("qualification_risk_repair_executed", False),
        ("scientific_checkpoint_published", True),
    ),
)
def test_v16_capacity_rank_fails_closed_when_workload_boundary_is_missing(
    field: str,
    value: object,
) -> None:
    with pytest.raises(Exception, match="rank evidence drifted"):
        replace(_rank(0), **{field: value}).validate()
