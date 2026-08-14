from __future__ import annotations

from pathlib import Path

import pytest

from rl_quant.training.top2000_m03r_v9_pretraining_step import model_state_sha256
from rl_quant.training.top2000_m03r_v12_initial_state import (
    M03RV12InitialStateError,
    load_m03r_v12_initial_parameter_state,
    write_m03r_v12_initial_parameter_state,
)
from rl_quant.training.top2000_m03r_v12_policy import (
    Top2000M03RV12PredictivePolicy,
)


def _policy(setting: int) -> Top2000M03RV12PredictivePolicy:
    return Top2000M03RV12PredictivePolicy(
        setting,
        selected_horizon_sessions=3,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )


def test_v12_initial_bytes_pair_settings_and_bind_selected_horizon(
    tmp_path: Path,
) -> None:
    source = _policy(0)
    path = tmp_path / "initial-state.pt"
    semantic_sha, file_sha = write_m03r_v12_initial_parameter_state(path, source)
    target = _policy(2)
    load_m03r_v12_initial_parameter_state(
        path,
        target,
        expected_file_sha256=file_sha,
        expected_state_sha256=semantic_sha,
    )
    assert model_state_sha256(target) == semantic_sha
    with pytest.raises(M03RV12InitialStateError, match="already exists"):
        write_m03r_v12_initial_parameter_state(path, source)
