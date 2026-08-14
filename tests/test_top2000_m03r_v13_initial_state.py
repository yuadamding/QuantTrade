from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rl_quant.training.top2000_m03r_v9_pretraining_step import model_state_sha256
from rl_quant.training.top2000_m03r_v13_initial_state import (
    M03RV13InitialStateError,
    load_m03r_v13_initial_parameter_state,
    write_m03r_v13_initial_parameter_state,
)
from rl_quant.training.top2000_m03r_v13_policy import (
    Top2000M03RV13PredictivePolicy,
)


def _policy(setting: int) -> Top2000M03RV13PredictivePolicy:
    return Top2000M03RV13PredictivePolicy(
        setting,
        selected_horizon_sessions=3,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )


def test_v13_common_initial_bytes_pair_both_settings(tmp_path: Path) -> None:
    torch.manual_seed(17)
    canonical = _policy(0)
    path = tmp_path / "common-v13-initial-state.pt"
    state_sha, file_sha, architecture_sha = write_m03r_v13_initial_parameter_state(
        path, canonical
    )

    torch.manual_seed(991)
    control = _policy(1)
    assert model_state_sha256(control) != state_sha
    load_m03r_v13_initial_parameter_state(
        path,
        control,
        expected_file_sha256=file_sha,
        expected_state_sha256=state_sha,
        expected_architecture_sha256=architecture_sha,
    )
    assert model_state_sha256(control) == state_sha
    assert control.v13_setting.setting_index == 1


def test_v13_common_initial_state_rejects_tampered_file(tmp_path: Path) -> None:
    torch.manual_seed(17)
    path = tmp_path / "common-v13-initial-state.pt"
    state_sha, file_sha, architecture_sha = write_m03r_v13_initial_parameter_state(
        path, _policy(0)
    )
    original = path.read_bytes()
    path.chmod(0o640)
    path.write_bytes(original + b"tamper")
    with pytest.raises(M03RV13InitialStateError, match="file hash"):
        load_m03r_v13_initial_parameter_state(
            path,
            _policy(1),
            expected_file_sha256=file_sha,
            expected_state_sha256=state_sha,
            expected_architecture_sha256=architecture_sha,
        )
