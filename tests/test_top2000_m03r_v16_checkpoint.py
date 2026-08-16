from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from rl_quant.training.top2000_m03r_v16_checkpoint import (
    M03RV16CheckpointError,
    load_m03r_v16_epoch_checkpoint_for_evaluation,
    write_immutable_m03r_v16_epoch_checkpoint,
    write_reload_evaluate_m03r_v16_epoch_checkpoint,
)
from rl_quant.training.top2000_m03r_v16_fold import (
    render_m03r_v16_fold_geometries,
)
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)


def _policy() -> Top2000M03RV16PredictivePolicy:
    return Top2000M03RV16PredictivePolicy(
        0,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )


def _identity() -> dict[str, object]:
    fold_index = 2
    epoch_index = 3
    return {
        "fold_index": fold_index,
        "epoch_index": epoch_index,
        "completed_score_updates": (
            render_m03r_v16_fold_geometries(1001)[fold_index].training_block_count
            * (epoch_index + 1)
        ),
        "panel_schedule_sha256": "a" * 64,
        "selection_target_operator_root_sha256": "b" * 64,
        "action_operator_root_sha256": "c" * 64,
        "source_array_sha256": "d" * 64,
        "asset_axis_sha256": "e" * 64,
    }


def test_v16_epoch_checkpoint_round_trip_uses_exact_loaded_bytes(
    tmp_path: Path,
) -> None:
    loaded, observed = write_reload_evaluate_m03r_v16_epoch_checkpoint(
        tmp_path / "epoch.pt",
        _policy(),
        _policy,
        lambda policy, receipt: (
            policy.v16_head_identity(),
            receipt.model_state_sha256,
        ),
        **_identity(),
    )
    assert observed[0] == loaded.head_identity
    assert observed[1] == loaded.model_state_sha256
    assert loaded.epoch_index == 3


def test_v16_epoch_checkpoint_rejects_wrong_update_cursor(tmp_path: Path) -> None:
    identity = _identity()
    identity["completed_score_updates"] = int(identity["completed_score_updates"]) - 1
    with pytest.raises(M03RV16CheckpointError, match="cursor"):
        write_immutable_m03r_v16_epoch_checkpoint(
            tmp_path / "epoch.pt",
            _policy(),
            **identity,  # type: ignore[arg-type]
        )


def test_v16_epoch_checkpoint_rejects_missing_selection_score_head(
    tmp_path: Path,
) -> None:
    clean = tmp_path / "clean.pt"
    write_immutable_m03r_v16_epoch_checkpoint(
        clean,
        _policy(),
        **_identity(),  # type: ignore[arg-type]
    )
    payload = torch.load(clean, map_location="cpu", weights_only=True)
    state = payload["model_state_dict"]
    del state["selection_score_head.bias"]
    from rl_quant.training.top2000_m03r_v9_pretraining_step import state_dict_sha256

    payload["model_state_sha256"] = state_dict_sha256(state)
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    identity = _identity()
    with pytest.raises(M03RV16CheckpointError, match="strict load"):
        load_m03r_v16_epoch_checkpoint_for_evaluation(
            tampered,
            expected_file_sha256=hashlib.sha256(tampered.read_bytes()).hexdigest(),
            expected_setting_index=0,
            expected_fold_index=int(identity["fold_index"]),
            expected_epoch_index=int(identity["epoch_index"]),
            expected_completed_score_updates=int(identity["completed_score_updates"]),
            expected_panel_schedule_sha256=str(identity["panel_schedule_sha256"]),
            expected_selection_target_operator_root_sha256=str(
                identity["selection_target_operator_root_sha256"]
            ),
            expected_action_operator_root_sha256=str(
                identity["action_operator_root_sha256"]
            ),
            expected_source_array_sha256=str(identity["source_array_sha256"]),
            expected_asset_axis_sha256=str(identity["asset_axis_sha256"]),
            policy=_policy(),
        )


def test_v16_epoch_checkpoint_rejects_hold_target_mismatch(tmp_path: Path) -> None:
    clean = tmp_path / "clean.pt"
    write_immutable_m03r_v16_epoch_checkpoint(
        clean,
        _policy(),
        **_identity(),  # type: ignore[arg-type]
    )
    payload = torch.load(clean, map_location="cpu", weights_only=True)
    payload["hold_target_sessions"] = 3
    tampered = tmp_path / "target-3.pt"
    torch.save(payload, tampered)
    identity = _identity()
    with pytest.raises(M03RV16CheckpointError, match="immutable identity"):
        load_m03r_v16_epoch_checkpoint_for_evaluation(
            tampered,
            expected_file_sha256=hashlib.sha256(tampered.read_bytes()).hexdigest(),
            expected_setting_index=0,
            expected_fold_index=int(identity["fold_index"]),
            expected_epoch_index=int(identity["epoch_index"]),
            expected_completed_score_updates=int(identity["completed_score_updates"]),
            expected_panel_schedule_sha256=str(identity["panel_schedule_sha256"]),
            expected_selection_target_operator_root_sha256=str(
                identity["selection_target_operator_root_sha256"]
            ),
            expected_action_operator_root_sha256=str(
                identity["action_operator_root_sha256"]
            ),
            expected_source_array_sha256=str(identity["source_array_sha256"]),
            expected_asset_axis_sha256=str(identity["asset_axis_sha256"]),
            policy=_policy(),
        )
