from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from rl_quant.training.top2000_m03r_v15_checkpoint import (
    M03RV15CheckpointError,
    load_m03r_v15_alpha_checkpoint_for_evaluation,
    write_immutable_m03r_v15_alpha_checkpoint,
    write_reload_evaluate_m03r_v15_checkpoint,
)
from rl_quant.training.top2000_m03r_v15_fold import render_m03r_v15_fold_geometries
from rl_quant.training.top2000_m03r_v15_policy import (
    Top2000M03RV15PredictivePolicy,
)


def _policy() -> Top2000M03RV15PredictivePolicy:
    return Top2000M03RV15PredictivePolicy(
        0,
        selected_horizon_sessions=3,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )


def _identity() -> dict[str, object]:
    return {
        "fold_index": 2,
        "completed_updates": render_m03r_v15_fold_geometries(1001)[
            2
        ].optimizer_updates,
        "selected_epoch_index": 3,
        "episode_schedule_sha256": "a" * 64,
        "target_residual_operator_root_sha256": "b" * 64,
        "action_residual_operator_root_sha256": "c" * 64,
        "source_array_sha256": "d" * 64,
        "asset_axis_sha256": "e" * 64,
        "checkpoint_selection_receipt_sha256": "f" * 64,
    }


def test_v15_checkpoint_round_trip_evaluates_exact_loaded_bytes(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    loaded, observed = write_reload_evaluate_m03r_v15_checkpoint(
        path,
        _policy(),
        _policy,
        lambda policy, receipt: (
            policy.v15_head_identity(),
            receipt.model_state_sha256,
        ),
        **_identity(),  # type: ignore[arg-type]
    )
    assert observed[0] == loaded.head_identity
    assert observed[1] == loaded.model_state_sha256
    assert loaded.selected_horizon_sessions == 3
    assert loaded.selected_epoch_index == 3
    assert (
        loaded.target_residual_operator_root_sha256
        != loaded.action_residual_operator_root_sha256
    )


def test_v15_checkpoint_rejects_wrong_fold_update_count(tmp_path: Path) -> None:
    identity = _identity()
    identity["completed_updates"] = int(identity["completed_updates"]) - 1
    with pytest.raises(M03RV15CheckpointError, match="cursor"):
        write_immutable_m03r_v15_alpha_checkpoint(
            tmp_path / "checkpoint.pt",
            _policy(),
            **identity,  # type: ignore[arg-type]
        )


def test_v15_checkpoint_rejects_missing_direct_mean_state(tmp_path: Path) -> None:
    clean = tmp_path / "clean.pt"
    file_sha = write_immutable_m03r_v15_alpha_checkpoint(
        clean,
        _policy(),
        **_identity(),  # type: ignore[arg-type]
    )
    assert len(file_sha) == 64
    payload = torch.load(clean, map_location="cpu", weights_only=True)
    state = payload["model_state_dict"]
    del state["economic_mean_head.bias"]
    from rl_quant.training.top2000_m03r_v9_pretraining_step import state_dict_sha256

    payload["model_state_sha256"] = state_dict_sha256(state)
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    tampered_sha = hashlib.sha256(tampered.read_bytes()).hexdigest()
    identity = _identity()
    with pytest.raises(M03RV15CheckpointError, match="strict load"):
        load_m03r_v15_alpha_checkpoint_for_evaluation(
            tampered,
            expected_file_sha256=tampered_sha,
            expected_setting_index=0,
            expected_fold_index=int(identity["fold_index"]),
            expected_completed_updates=int(identity["completed_updates"]),
            expected_selected_epoch_index=int(identity["selected_epoch_index"]),
            expected_episode_schedule_sha256=str(
                identity["episode_schedule_sha256"]
            ),
            expected_target_residual_operator_root_sha256=str(
                identity["target_residual_operator_root_sha256"]
            ),
            expected_action_residual_operator_root_sha256=str(
                identity["action_residual_operator_root_sha256"]
            ),
            expected_source_array_sha256=str(identity["source_array_sha256"]),
            expected_asset_axis_sha256=str(identity["asset_axis_sha256"]),
            expected_checkpoint_selection_receipt_sha256=str(
                identity["checkpoint_selection_receipt_sha256"]
            ),
            policy=_policy(),
        )
