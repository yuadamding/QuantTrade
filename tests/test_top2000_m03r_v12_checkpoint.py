from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from rl_quant.training.top2000_m03r_v12_checkpoint import (
    M03RV12CheckpointError,
    load_m03r_v12_alpha_checkpoint_for_evaluation,
    write_immutable_m03r_v12_alpha_checkpoint,
    write_reload_evaluate_m03r_v12_checkpoint,
)
from rl_quant.training.top2000_m03r_v12_policy import (
    Top2000M03RV12PredictivePolicy,
)


def _policy() -> Top2000M03RV12PredictivePolicy:
    return Top2000M03RV12PredictivePolicy(
        1,
        selected_horizon_sessions=3,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )


def _identity() -> dict[str, object]:
    return {
        "fold_index": 2,
        "selected_horizon_sessions": 3,
        "episode_schedule_sha256": "a" * 64,
        "residual_operator_root_sha256": "b" * 64,
        "source_array_sha256": "c" * 64,
        "asset_axis_sha256": "d" * 64,
    }


def test_v12_checkpoint_round_trip_binds_all_three_heads(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    loaded, observed = write_reload_evaluate_m03r_v12_checkpoint(
        path,
        _policy(),
        _policy,
        lambda policy, receipt: (
            policy.v12_head_identity(),
            receipt.model_state_sha256,
        ),
        **_identity(),  # type: ignore[arg-type]
    )
    assert observed[0] == loaded.head_identity
    assert observed[1] == loaded.model_state_sha256
    assert (
        len(
            {
                loaded.head_identity.economic_mean_head_state_sha256,
                loaded.head_identity.economic_scale_head_state_sha256,
                loaded.head_identity.rank_score_head_state_sha256,
            }
        )
        == 3
    )


def test_v12_checkpoint_rejects_horizon_drift(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    file_sha = write_immutable_m03r_v12_alpha_checkpoint(
        path,
        _policy(),
        **_identity(),  # type: ignore[arg-type]
    )
    with pytest.raises(M03RV12CheckpointError, match="horizon"):
        load_m03r_v12_alpha_checkpoint_for_evaluation(
            path,
            expected_file_sha256=file_sha,
            expected_setting_index=1,
            expected_fold_index=2,
            expected_selected_horizon_sessions=30,
            expected_episode_schedule_sha256="a" * 64,
            expected_residual_operator_root_sha256="b" * 64,
            expected_source_array_sha256="c" * 64,
            expected_asset_axis_sha256="d" * 64,
            policy=_policy(),
        )


def test_v12_checkpoint_rejects_missing_rank_head_buffer(tmp_path: Path) -> None:
    clean = tmp_path / "clean.pt"
    write_immutable_m03r_v12_alpha_checkpoint(
        clean,
        _policy(),
        **_identity(),  # type: ignore[arg-type]
    )
    payload = torch.load(clean, map_location="cpu", weights_only=True)
    state = payload["model_state_dict"]
    del state["rank_score_head.bias"]
    from rl_quant.training.top2000_m03r_v9_pretraining_step import state_dict_sha256

    payload["model_state_sha256"] = state_dict_sha256(state)
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    file_sha = hashlib.sha256(tampered.read_bytes()).hexdigest()
    with pytest.raises(M03RV12CheckpointError, match="strict load"):
        load_m03r_v12_alpha_checkpoint_for_evaluation(
            tampered,
            expected_file_sha256=file_sha,
            expected_setting_index=1,
            expected_fold_index=2,
            expected_selected_horizon_sessions=3,
            expected_episode_schedule_sha256="a" * 64,
            expected_residual_operator_root_sha256="b" * 64,
            expected_source_array_sha256="c" * 64,
            expected_asset_axis_sha256="d" * 64,
            policy=_policy(),
        )
