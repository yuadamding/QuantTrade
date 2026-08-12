from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import M03RV9HorizonBinding
from rl_quant.training.top2000_m03r_v9_policy import Top2000M03RV9PredictivePolicy
from rl_quant.training.top2000_m03r_v9_pretraining_step import state_dict_sha256
from rl_quant.training.top2000_m03r_v11_checkpoint import (
    M03RV11CheckpointError,
    load_m03r_v11_alpha_checkpoint_for_evaluation,
    write_immutable_m03r_v11_alpha_checkpoint,
    write_reload_evaluate_m03r_v11_checkpoint,
)


def _policy() -> Top2000M03RV9PredictivePolicy:
    return Top2000M03RV9PredictivePolicy(
        0,
        M03RV9HorizonBinding(30, 30, 30),
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_evaluation_uses_exact_round_tripped_checkpoint_bytes(tmp_path: Path) -> None:
    policy = _policy()
    expected_state = state_dict_sha256(policy.state_dict())
    loaded, evaluated = write_reload_evaluate_m03r_v11_checkpoint(
        tmp_path / "v11.pt",
        policy,
        _policy,
        lambda loaded_policy, receipt: (
            state_dict_sha256(loaded_policy.state_dict()),
            receipt.checkpoint_file_sha256,
        ),
        setting_index=1,
        fold_index=2,
        selected_horizon_sessions=30,
        episode_schedule_sha256="a" * 64,
        residual_operator_root_sha256="b" * 64,
        source_array_sha256="c" * 64,
        asset_axis_sha256="d" * 64,
    )
    assert loaded.model_state_sha256 == expected_state
    assert evaluated == (expected_state, loaded.checkpoint_file_sha256)


def test_missing_serialized_buffer_fails_strict_load(tmp_path: Path) -> None:
    path = tmp_path / "valid.pt"
    file_sha = write_immutable_m03r_v11_alpha_checkpoint(
        path,
        _policy(),
        setting_index=1,
        fold_index=0,
        selected_horizon_sessions=30,
        episode_schedule_sha256="a" * 64,
        residual_operator_root_sha256="b" * 64,
        source_array_sha256="c" * 64,
        asset_axis_sha256="d" * 64,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model_state_dict"]
    state.pop(next(iter(state)))
    payload["model_state_sha256"] = state_dict_sha256(state)
    broken = tmp_path / "broken.pt"
    torch.save(payload, broken)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_m03r_v11_alpha_checkpoint_for_evaluation(
            broken,
            expected_file_sha256=_hash(broken),
            expected_setting_index=1,
            expected_fold_index=0,
            expected_selected_horizon_sessions=30,
            expected_episode_schedule_sha256="a" * 64,
            expected_residual_operator_root_sha256="b" * 64,
            expected_source_array_sha256="c" * 64,
            expected_asset_axis_sha256="d" * 64,
            policy=_policy(),
        )
    with pytest.raises(M03RV11CheckpointError, match="file hash"):
        load_m03r_v11_alpha_checkpoint_for_evaluation(
            path,
            expected_file_sha256="e" * 64,
            expected_setting_index=1,
            expected_fold_index=0,
            expected_selected_horizon_sessions=30,
            expected_episode_schedule_sha256="a" * 64,
            expected_residual_operator_root_sha256="b" * 64,
            expected_source_array_sha256="c" * 64,
            expected_asset_axis_sha256="d" * 64,
            policy=_policy(),
        )
    assert len(file_sha) == 64
