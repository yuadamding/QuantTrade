"""Focused tests for the frozen seed-17 2026 checkpoint loader."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from rl_quant.evaluation.top2000_m03r_v7_2026_checkpoint import (
    Top2000M03RV72026CheckpointError,
    load_top2000_m03r_v7_seed17_2026_checkpoint,
)
from rl_quant.evaluation.top2000_m03r_v7_dev import model_state_sha256
from rl_quant.protocol.hold30_alpha_m03r_v7_seed17_top2000_2026_ytd import (
    M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS,
    Top2000M03RV7DevelopmentPolicy,
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.workflows.top2000_m03r_v7_dev import (
    CELL_MODEL_SCHEMA,
    optimizer_state_dict_sha256,
)
from rl_quant.workflows.top2000_m03r_v7_seed17_2026_ytd import (
    Top2000M03RV7Seed172026YTDCheckpointBinding,
)
from rl_quant.workflows.top2000_m03r_v7_seed17_dev import (
    Top2000M03RV7Seed17TrainingPlan,
)


def _digest(value: str | bytes) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    return _digest(path.read_bytes())


def _write_json(path: Path, payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return _digest(encoded)


def _checkpoint_fixture(
    tmp_path: Path,
    *,
    world_size: int = 2,
    completed_optimizer_steps: int = 64,
) -> tuple[Path, Top2000M03RV7Seed172026YTDCheckpointBinding]:
    contract = M03R_SEED17_TOP2000_2026_YTD_EVALUATION_CONTRACT
    setting = contract.settings[0]
    root = tmp_path / "training-output"
    setting_root = root / "completion-00-setting-00"
    training_root = setting_root / "training"
    plan = Top2000M03RV7Seed17TrainingPlan(
        setting_index=0,
        setting_id=setting.seed17_setting_id,
        runtime_setting_id=setting.runtime_setting_id,
        cache_path=str(tmp_path / "cache.pt"),
        cache_sha256=_digest("cache"),
        output_root=str(setting_root),
    )
    plan_path = setting_root / "training-plan.json"
    plan_file_sha256 = _write_json(plan_path, asdict(plan))
    fold = render_top2000_m03r_v7_development_folds(
        TOP2000_M03R_V7_DEV_REQUIRED_STATE_ROWS
    )[5]
    torch.manual_seed(17)
    policy = Top2000M03RV7DevelopmentPolicy(
        setting.runtime_setting_id,
        token_dim=plan.token_dim,
        raw_stock_chunk=plan.raw_stock_chunk,
        activation_checkpointing=plan.activation_checkpointing,
    )
    state_sha256 = model_state_sha256(policy)
    optimizer_state: dict[str, object] = {}
    model_path = training_root / "cells/fold-05-seed-17/model.rank-00.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": CELL_MODEL_SCHEMA,
            "protocol_sha256": plan.protocol_sha256,
            "plan_file_sha256": plan_file_sha256,
            "plan_receipt_sha256": plan.receipt_sha256,
            "cache_sha256": plan.cache_sha256,
            "setting_index": plan.setting_index,
            "setting_id": plan.setting_id,
            "fold_index": fold.fold_index,
            "fold_receipt_sha256": fold.receipt_sha256,
            "seed": 17,
            "rank": 0,
            "world_size": world_size,
            "completed_optimizer_steps": completed_optimizer_steps,
            "model_state_dict": policy.state_dict(),
            "overlay_optimizer_required": False,
            "alpha_core_optimizer_state_dict": optimizer_state,
            "alpha_core_optimizer_state_sha256": (
                optimizer_state_dict_sha256(optimizer_state)
            ),
            "overlay_optimizer_state_dict": None,
            "overlay_optimizer_state_sha256": None,
            "development_only": True,
            "promotion_eligible": False,
        },
        model_path,
    )
    model_file_sha256 = _file_sha256(model_path)
    return root, Top2000M03RV7Seed172026YTDCheckpointBinding(
        completion_index=0,
        setting_index=0,
        setting_id=setting.seed17_setting_id,
        runtime_setting_id=setting.runtime_setting_id,
        training_fold_index=5,
        seed=17,
        writer_rank=0,
        optimizer_steps=64,
        checkpoint_role="headline",
        training_root_relative_path="completion-00-setting-00/training",
        model_relative_path=(
            "completion-00-setting-00/training/cells/"
            "fold-05-seed-17/model.rank-00.pt"
        ),
        model_file_sha256=model_file_sha256,
        model_state_sha256=state_sha256,
        cell_receipt_relative_path="completion-00-setting-00/training/cell.json",
        cell_receipt_file_sha256=_digest("cell"),
        seed_validation_receipt_relative_path=(
            "completion-00-setting-00/training/validation.json"
        ),
        seed_validation_receipt_file_sha256=_digest("validation"),
        fold_execution_receipt_relative_path=(
            "completion-00-setting-00/training/execution.json"
        ),
        fold_execution_receipt_file_sha256=_digest("execution"),
        completion_receipt_relative_path=(
            "completion-00-setting-00/training/completion.json"
        ),
        completion_receipt_file_sha256=_digest("completion"),
        training_plan_file_sha256=plan_file_sha256,
        training_plan_receipt_sha256=plan.receipt_sha256,
    )


def test_loader_reconstructs_exact_plan_fold_and_inference_only_policy(
    tmp_path: Path,
) -> None:
    root, binding = _checkpoint_fixture(tmp_path)

    loaded = load_top2000_m03r_v7_seed17_2026_checkpoint(
        binding,
        training_output_root=root,
        device="cpu",
    )

    assert loaded.training_plan.receipt_sha256 == binding.training_plan_receipt_sha256
    assert loaded.training_fold.fold_index == 5
    assert loaded.receipt.frozen_checkpoint_binding_sha256 == binding.receipt_sha256
    assert loaded.receipt.world_size == 2
    assert loaded.receipt.completed_optimizer_steps == 64
    assert loaded.receipt.checkpoint_role == "headline"
    assert model_state_sha256(loaded.policy) == binding.model_state_sha256
    assert not loaded.policy.training
    assert not any(parameter.requires_grad for parameter in loaded.policy.parameters())
    assert not loaded.receipt.policy_training_enabled
    assert not loaded.receipt.scientific_reporting_eligible
    assert not loaded.receipt.promotion_eligible


@pytest.mark.parametrize(
    ("world_size", "completed_steps"),
    ((1, 64), (2, 63)),
)
def test_loader_rejects_nonqualified_world_size_or_incomplete_updates(
    tmp_path: Path,
    world_size: int,
    completed_steps: int,
) -> None:
    root, binding = _checkpoint_fixture(
        tmp_path,
        world_size=world_size,
        completed_optimizer_steps=completed_steps,
    )

    with pytest.raises(Top2000M03RV72026CheckpointError, match="two-rank 64-update"):
        load_top2000_m03r_v7_seed17_2026_checkpoint(
            binding,
            training_output_root=root,
            device="cpu",
        )


def test_loader_rejects_plan_and_model_state_binding_drift(tmp_path: Path) -> None:
    root, binding = _checkpoint_fixture(tmp_path)
    changed_state_binding = replace(binding, model_state_sha256=_digest("wrong-state"))
    with pytest.raises(Top2000M03RV72026CheckpointError, match="model-state"):
        load_top2000_m03r_v7_seed17_2026_checkpoint(
            changed_state_binding,
            training_output_root=root,
            device="cpu",
        )

    plan_path = root / "completion-00-setting-00/training-plan.json"
    payload = json.loads(plan_path.read_bytes())
    payload["cache_path"] = str(tmp_path / "different-cache.pt")
    changed_file_sha256 = _write_json(plan_path, payload)
    changed_plan_binding = replace(
        binding,
        training_plan_file_sha256=changed_file_sha256,
    )
    with pytest.raises(Top2000M03RV72026CheckpointError, match="training plan"):
        load_top2000_m03r_v7_seed17_2026_checkpoint(
            changed_plan_binding,
            training_output_root=root,
            device="cpu",
        )
