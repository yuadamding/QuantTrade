from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import pytest

from rl_quant.protocol.hold30 import HOLD30_PROTOCOL_GENERATION
from rl_quant.protocol.hold30_freeze import (
    HOLD30_MIN_AXIS_POSITIONS,
    Hold30FreezeBindings,
    Hold30FreezeError,
    hold30_trial_inventory,
    render_hold30_folds,
    render_hold30_manifest,
    sha256_payload,
)


def _axis(count: int = HOLD30_MIN_AXIS_POSITIONS) -> tuple[str, ...]:
    start = date(2018, 1, 1)
    return tuple((start + timedelta(days=index)).isoformat() for index in range(count))


def _bindings(axis: tuple[str, ...]) -> Hold30FreezeBindings:
    split = [fold.__dict__ if hasattr(fold, "__dict__") else {
        field: getattr(fold, field) for field in fold.__dataclass_fields__
    } for fold in render_hold30_folds(axis)]
    digest = "1" * 64
    return Hold30FreezeBindings(
        repository_url="ssh://example/QuantTrade.git",
        git_commit="2" * 40,
        git_tree="3" * 40,
        clean_worktree=True,
        dirty_patch_sha256=None,
        source_archive_sha256=digest,
        dependency_lock_sha256=digest,
        container_image_digest="sha256:" + digest,
        rfc_sha256=digest,
        base_experiment_sha256=digest,
        v2_specification_sha256=digest,
        data_snapshot_sha256=digest,
        decision_axis_sha256=sha256_payload(axis),
        universe_events_sha256=digest,
        corporate_actions_sha256=digest,
        benchmark_trace_sha256=digest,
        split_arrays_sha256=sha256_payload(split),
        component_qualification_sha256=digest,
        software_qualification_sha256=digest,
        data_qualification_sha256=digest,
        capacity_qualification_sha256=digest,
        training_plan_sha256=digest,
        stage1_plan_sha256=digest,
        control_plan_sha256=digest,
        inference_plan_sha256=digest,
        artifact_inventory_sha256=digest,
        recovery_policy_sha256=digest,
        worker_template_sha256=digest,
        admitted_job_template_sha256=digest,
        namespace="yn-gpu-workload",
        service_account="hold30-runner",
    )


def test_fold_renderer_freezes_six_exact_prelockbox_partitions() -> None:
    folds = render_hold30_folds(_axis())
    assert len(folds) == 6
    assert folds[0].expanding_train == (0, 472)
    assert folds[0].training_anchors == (63, 441)
    assert folds[-1].outer_support[1] == HOLD30_MIN_AXIS_POSITIONS
    assert folds[-1].embargo == (HOLD30_MIN_AXIS_POSITIONS, HOLD30_MIN_AXIS_POSITIONS)


def test_fold_renderer_fails_closed_on_short_duplicate_or_post2025_axis() -> None:
    axis = _axis()
    with pytest.raises(Hold30FreezeError, match="N >= 1811"):
        render_hold30_folds(axis[:-1])
    duplicate = axis[:100] + (axis[99],) + axis[101:]
    with pytest.raises(Hold30FreezeError, match="strictly increasing"):
        render_hold30_folds(duplicate)
    shifted = tuple(
        (date(2022, 1, 1) + timedelta(days=index)).isoformat()
        for index in range(HOLD30_MIN_AXIS_POSITIONS)
    )
    with pytest.raises(Hold30FreezeError, match="before 2026"):
        render_hold30_folds(shifted)


def test_inventory_is_exactly_eight_by_six_by_five() -> None:
    inventory = hold30_trial_inventory()
    assert len(inventory) == 240
    assert len({row["setting_id"] for row in inventory}) == 8
    assert len({row["fold_index"] for row in inventory}) == 6
    assert {row["seed"] for row in inventory} == {17, 29, 43, 71, 101}


def test_manifest_binds_axis_splits_and_exact_16_h100_shape() -> None:
    axis = _axis()
    manifest = render_hold30_manifest(axis, _bindings(axis), approval_state="software_qualified")
    assert manifest["protocol_generation"] == HOLD30_PROTOCOL_GENERATION
    assert manifest["trial_inventory_count"] == 240
    assert manifest["compute"] == {
        "gpu_product": "NVIDIA-H100-80GB-HBM3",
        "gpus_per_setting": 2,
        "world_size_per_trial": 2,
        "distributed_strategy": "explicit-sum-origin-shard-two-rank-v1",
        "concurrent_setting_workers": 8,
        "maximum_h100": 16,
        "namespace": "yn-gpu-workload",
        "service_account": "hold30-runner",
        "worker_template_sha256": "1" * 64,
        "admitted_job_template_sha256": "1" * 64,
        "scientific_fields_inferred_from_gpu_count": False,
    }
    assert set(manifest["plans"]) == {
        "training_plan_sha256",
        "stage1_plan_sha256",
        "control_plan_sha256",
        "inference_plan_sha256",
        "artifact_inventory_sha256",
        "recovery_policy_sha256",
    }
    assert manifest["render_grants_launch_authority"] is False
    assert len(manifest["manifest_sha256"]) == 64


def test_manifest_rejects_stale_axis_or_split_digest() -> None:
    axis = _axis()
    bindings = _bindings(axis)
    with pytest.raises(Hold30FreezeError, match="decision-axis digest"):
        render_hold30_manifest(axis, replace(bindings, decision_axis_sha256="0" * 64))
    with pytest.raises(Hold30FreezeError, match="split-arrays digest"):
        render_hold30_manifest(axis, replace(bindings, split_arrays_sha256="0" * 64))


def test_dirty_source_requires_a_patch_digest_and_image_requires_a_digest() -> None:
    axis = _axis()
    bindings = _bindings(axis)
    with pytest.raises(Hold30FreezeError, match="dirty worktree"):
        replace(bindings, clean_worktree=False)
    with pytest.raises(Hold30FreezeError, match="digest-pinned"):
        replace(bindings, container_image_digest="latest")


def test_executable_state_requires_a_separate_approval_receipt() -> None:
    axis = _axis()
    bindings = _bindings(axis)
    with pytest.raises(Hold30FreezeError, match="executable_approval_sha256"):
        render_hold30_manifest(axis, bindings, approval_state="executable")
    approved = replace(bindings, executable_approval_sha256="4" * 64)
    manifest = render_hold30_manifest(axis, approved, approval_state="executable")
    assert manifest["approval_state"] == "executable"
    with pytest.raises(Hold30FreezeError, match="non-executable"):
        render_hold30_manifest(axis, approved, approval_state="dry_run")
