from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rl_quant.training.top2000_m03r_v9_pretraining_step import model_state_sha256
from rl_quant.training.top2000_m03r_v16_initial_state import (
    M03RV16InitialStateError,
    load_m03r_v16_initial_parameter_state,
    write_m03r_v16_initial_parameter_state,
)
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)


def _policy(setting: int) -> Top2000M03RV16PredictivePolicy:
    return Top2000M03RV16PredictivePolicy(
        setting,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )


def test_v16_common_initial_bytes_pair_all_settings(tmp_path: Path) -> None:
    torch.manual_seed(17)
    canonical = _policy(0)
    path = tmp_path / "common-v16-initial-state.pt"
    state_sha, file_sha, architecture_sha = write_m03r_v16_initial_parameter_state(
        path, canonical
    )
    for setting in (1, 2):
        torch.manual_seed(900 + setting)
        control = _policy(setting)
        assert model_state_sha256(control) != state_sha
        load_m03r_v16_initial_parameter_state(
            path,
            control,
            expected_file_sha256=file_sha,
            expected_state_sha256=state_sha,
            expected_architecture_sha256=architecture_sha,
        )
        assert model_state_sha256(control) == state_sha
        assert control.v16_setting.setting_index == setting


def test_v16_common_initial_state_rejects_tampered_file(tmp_path: Path) -> None:
    torch.manual_seed(17)
    path = tmp_path / "common-v16-initial-state.pt"
    state_sha, file_sha, architecture_sha = write_m03r_v16_initial_parameter_state(
        path, _policy(0)
    )
    original = path.read_bytes()
    path.chmod(0o640)
    path.write_bytes(original + b"tamper")
    with pytest.raises(M03RV16InitialStateError, match="file hash"):
        load_m03r_v16_initial_parameter_state(
            path,
            _policy(2),
            expected_file_sha256=file_sha,
            expected_state_sha256=state_sha,
            expected_architecture_sha256=architecture_sha,
        )
