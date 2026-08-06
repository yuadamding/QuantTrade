"""Artifact-backed tests for the TOP2000 two-H100 qualification gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from rl_quant.training.hold30_alpha_m03r_v7_package import (
    M03RV7Top2000ArtifactBindings,
    M03RV7Top2000PackageError,
    M03RV7Top2000QualifiedPackage,
    build_m03r_v7_top2000_package_plan,
    build_m03r_v7_top2000_worker_receipt,
    verify_m03r_v7_top2000_qualification_artifact,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> str:
    encoded = (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _plan() -> Any:
    image = _digest("image")
    return build_m03r_v7_top2000_package_plan(
        artifacts=M03RV7Top2000ArtifactBindings(
            source_archive_sha256=_digest("source archive"),
            source_manifest_sha256=_digest("source manifest"),
            dependency_lock_sha256=_digest("dependencies"),
            cache_artifact_sha256=_digest("cache"),
            cache_manifest_sha256=_digest("cache manifest"),
            data_manifest_sha256=_digest("data manifest"),
            execution_model_sha256=_digest("execution"),
            image_reference=f"registry/research@sha256:{image}",
            image_digest_sha256=image,
        ),
        plan_artifact_path="/mnt/package/package-plan.json",
    )


def _rank(
    rank: int,
    *,
    allocated_gib: float = 50.0,
    reserved_gib: float = 65.0,
    measured_bytes: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    if measured_bytes is None:
        allocated_bytes = int((allocated_gib + rank * 0.25) * 1024**3)
        reserved_bytes = int((reserved_gib + rank * 0.25) * 1024**3)
        total_memory_bytes = 80 * 1024**3
    else:
        allocated_bytes, reserved_bytes, total_memory_bytes = measured_bytes
    return {
        "rank": rank,
        "device": f"cuda:{rank}",
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "gpu_total_memory_bytes": total_memory_bytes,
        "compute_capability": [9, 0],
        "peak_allocated_bytes": allocated_bytes,
        "peak_reserved_bytes": reserved_bytes,
        "allocator_oom_count": 0,
        "allocator_retry_count": 0,
        "torchrun_restart_count": 1,
        "python_version": "3.11.9",
        "torch_version": "2.5.1",
        "torch_cuda_version": "12.4",
        "cudnn_version": 90100,
        "nccl_version": [2, 21, 5],
    }


def _qualification_tree(
    tmp_path: Path,
    *,
    allocated_gib: float = 50.0,
    reserved_gib: float = 65.0,
    measured_rank_bytes: tuple[tuple[int, int, int], tuple[int, int, int]]
    | None = None,
    binding_activation_checkpointing: bool | None = None,
) -> tuple[Any, Path, str, Path]:
    plan = _plan()
    row = plan.indices[0]
    setting_root = tmp_path / "completion-00-setting-00"
    training_plan = {
        "setting_index": row.setting_index,
        "setting_id": row.development_setting_id,
        "cache_sha256": plan.artifacts.cache_artifact_sha256,
        "expected_world_size": 2,
        "token_dim": plan.runtime_profile.token_dim,
        "max_origin_batch": plan.runtime_profile.max_origin_batch,
        "activation_checkpointing": (
            plan.runtime_profile.activation_checkpointing
            if binding_activation_checkpointing is None
            else binding_activation_checkpointing
        ),
    }
    binding = {
        "schema": "rl-quant.top2000-dev.m03r-v7-package-worker-binding-v1",
        "package_plan_sha256": plan.package_plan_sha256,
        "completion": {
            "completion_index": 0,
            "setting_index": row.setting_index,
            "development_setting_id": row.development_setting_id,
        },
        "training_plan": training_plan,
    }
    binding_sha = _write(setting_root / "execution-plan-binding.json", binding)
    model_hash = _digest("model state")
    optimizer_hash = _digest("optimizer state")
    model_file_hashes: list[str] = []
    for rank in range(2):
        model_path = (
            setting_root
            / "qualification"
            / "cells"
            / "fold-00-seed-17"
            / f"model.rank-{rank:02d}.pt"
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(f"model-{rank}".encode())
        model_file_hashes.append(hashlib.sha256(model_path.read_bytes()).hexdigest())
    peaks = [
        _rank(
            rank,
            allocated_gib=allocated_gib,
            reserved_gib=reserved_gib,
            measured_bytes=(
                None if measured_rank_bytes is None else measured_rank_bytes[rank]
            ),
        )
        for rank in range(2)
    ]
    cell = {
        "schema": "rl-quant.top2000-dev.m03r-v7-cell-receipt-v2",
        "mode": "qualification",
        "protocol_sha256": plan.protocol_sha256,
        "plan_file_sha256": binding_sha,
        "cache_sha256": plan.artifacts.cache_artifact_sha256,
        "setting_index": row.setting_index,
        "setting_id": row.development_setting_id,
        "optimizer_steps": 4,
        "fold_index": 0,
        "seed": 17,
        "rank_model_sha256": model_file_hashes,
        "rank_model_state_sha256": [model_hash, model_hash],
        "rank_alpha_core_optimizer_state_sha256": [optimizer_hash, optimizer_hash],
        "rank_overlay_optimizer_state_sha256": [None, None],
        "rank_peak_cuda_memory": peaks,
        "last_metrics": {"objective": 0.125},
        "seed_validation_required": False,
    }
    receipt_root = setting_root / "qualification"
    cell_path = receipt_root / "receipts" / "fold-00-seed-17.json"
    cell_sha = _write(cell_path, cell)
    terminal = {
        "schema": "rl-quant.top2000-dev.m03r-v7-bounded-qualification-v1",
        "mode": "qualification",
        "protocol_sha256": plan.protocol_sha256,
        "plan_file_sha256": binding_sha,
        "cache_sha256": plan.artifacts.cache_artifact_sha256,
        "setting_index": row.setting_index,
        "setting_id": row.development_setting_id,
        "world_size": 2,
        "fold_count": 1,
        "paired_seeds": [17],
        "completed_cells": 1,
        "optimizer_steps_per_cell": 4,
        "intentional_restart_after_step": 1,
        "resumed_from_checkpoint": True,
        "resume_completed_steps": 1,
        "seed_validation_receipt_count": 0,
        "fold_ensemble_receipt_count": 0,
        "inference_path_count": 0,
        "output_space_ensemble_required": False,
        "development_only": True,
        "future_selected_universe": True,
        "outer_evaluation_authorized": False,
        "promotion_eligible": False,
        "complete": True,
        "cell_receipt_sha256": {cell_path.name: cell_sha},
        "rank_peak_cuda_memory": peaks,
        "rank_elapsed_seconds": [11.0, 11.2],
        "rank_model_state_sha256": [model_hash, model_hash],
        "rank_alpha_core_optimizer_state_sha256": [optimizer_hash, optimizer_hash],
        "rank_overlay_optimizer_state_sha256": [None, None],
    }
    terminal_path = receipt_root / "qualification-receipt.json"
    terminal_sha = _write(terminal_path, terminal)
    return plan, terminal_path, terminal_sha, cell_path


def test_verifier_derives_real_two_rank_memory_parity_and_restart(
    tmp_path: Path,
) -> None:
    plan, receipt, receipt_sha, _ = _qualification_tree(tmp_path)
    verified = verify_m03r_v7_top2000_qualification_artifact(
        plan=plan,
        completion_index=0,
        qualification_receipt_path=receipt,
        expected_qualification_receipt_sha256=receipt_sha,
    )
    assert verified.rank_peak_allocated_bytes == (
        50 * 1024**3,
        int(50.25 * 1024**3),
    )
    assert verified.rank_peak_reserved_bytes == (
        65 * 1024**3,
        int(65.25 * 1024**3),
    )
    assert len(set(verified.rank_model_state_sha256)) == 1
    assert verified.qualification_steps == 4


def test_verifier_accepts_exact_a9_allocator_residency_bytes(tmp_path: Path) -> None:
    measured = (
        (53_888_892_928, 69_986_156_544, 85_028_372_480),
        (53_628_473_344, 69_736_595_456, 85_028_372_480),
    )
    plan, receipt, receipt_sha, _ = _qualification_tree(
        tmp_path,
        measured_rank_bytes=measured,
    )
    verified = verify_m03r_v7_top2000_qualification_artifact(
        plan=plan,
        completion_index=0,
        qualification_receipt_path=receipt,
        expected_qualification_receipt_sha256=receipt_sha,
    )
    assert verified.rank_peak_allocated_bytes == tuple(value[0] for value in measured)
    assert verified.rank_peak_reserved_bytes == tuple(value[1] for value in measured)
    assert verified.rank_total_memory_bytes == tuple(value[2] for value in measured)


def test_verifier_rejects_tampered_cell_and_underfilled_memory(tmp_path: Path) -> None:
    plan, receipt, receipt_sha, cell_path = _qualification_tree(tmp_path / "tamper")
    cell_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(M03RV7Top2000PackageError, match="SHA-256 mismatch"):
        verify_m03r_v7_top2000_qualification_artifact(
            plan=plan,
            completion_index=0,
            qualification_receipt_path=receipt,
            expected_qualification_receipt_sha256=receipt_sha,
        )

    plan, receipt, receipt_sha, _ = _qualification_tree(
        tmp_path / "checkpointing-drift",
        binding_activation_checkpointing=True,
    )
    with pytest.raises(
        M03RV7Top2000PackageError,
        match="execution-plan binding",
    ):
        verify_m03r_v7_top2000_qualification_artifact(
            plan=plan,
            completion_index=0,
            qualification_receipt_path=receipt,
            expected_qualification_receipt_sha256=receipt_sha,
        )

    plan, receipt, receipt_sha, _ = _qualification_tree(
        tmp_path / "small",
        allocated_gib=20,
    )
    with pytest.raises(M03RV7Top2000PackageError, match="useful allocated"):
        verify_m03r_v7_top2000_qualification_artifact(
            plan=plan,
            completion_index=0,
            qualification_receipt_path=receipt,
            expected_qualification_receipt_sha256=receipt_sha,
        )

    plan, receipt, receipt_sha, _ = _qualification_tree(
        tmp_path / "allocator-padding",
        allocated_gib=48,
        reserved_gib=70,
    )
    with pytest.raises(M03RV7Top2000PackageError, match="ratio of at least 0.70"):
        verify_m03r_v7_top2000_qualification_artifact(
            plan=plan,
            completion_index=0,
            qualification_receipt_path=receipt,
            expected_qualification_receipt_sha256=receipt_sha,
        )

    plan, receipt, receipt_sha, _ = _qualification_tree(
        tmp_path / "under-resident",
        allocated_gib=48,
        reserved_gib=59,
    )
    with pytest.raises(M03RV7Top2000PackageError, match="allocator-reserved HBM"):
        verify_m03r_v7_top2000_qualification_artifact(
            plan=plan,
            completion_index=0,
            qualification_receipt_path=receipt,
            expected_qualification_receipt_sha256=receipt_sha,
        )


def test_caller_authored_worker_draft_cannot_enter_qualified_package() -> None:
    plan = _plan()
    draft = build_m03r_v7_top2000_worker_receipt(
        plan=plan,
        worker_argv_prefix=("/opt/conda/envs/quanttrade/bin/python",),
        worker_entrypoint_sha256=_digest("worker"),
        runtime_manifest_sha256=_digest("runtime"),
        smoke_test_receipt_sha256=_digest("smoke"),
        cuda_two_rank_parity_receipt_sha256=_digest("parity"),
        exact_restart_receipt_sha256=_digest("restart"),
    )
    with pytest.raises(M03RV7Top2000PackageError, match="caller-authored"):
        M03RV7Top2000QualifiedPackage(plan=plan, worker_receipt=draft)
