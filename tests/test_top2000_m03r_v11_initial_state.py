from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from rl_quant.training.top2000_m03r_v9_pretraining_step import model_state_sha256
from rl_quant.training.top2000_m03r_v11_initial_state import (
    M03RV11InitialStateError,
    load_m03r_v11_initial_parameter_state,
    write_m03r_v11_initial_parameter_state,
)


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(5, 3)
        self.register_buffer("running_scale", torch.arange(3, dtype=torch.float32))


def test_packaged_initial_state_replaces_fresh_runtime_initialization(
    tmp_path: Path,
) -> None:
    torch.manual_seed(17)
    package_policy = _Policy()
    path = tmp_path / "common-initial-parameter-state.pt"
    state_sha, file_sha = write_m03r_v11_initial_parameter_state(path, package_policy)

    torch.manual_seed(918273)
    runtime_policy = _Policy()
    assert model_state_sha256(runtime_policy) != state_sha
    load_m03r_v11_initial_parameter_state(
        path,
        runtime_policy,
        expected_file_sha256=file_sha,
        expected_state_sha256=state_sha,
    )
    assert model_state_sha256(runtime_policy) == state_sha


def test_packaged_initial_state_rejects_file_and_shape_drift(tmp_path: Path) -> None:
    torch.manual_seed(17)
    path = tmp_path / "common-initial-parameter-state.pt"
    state_sha, file_sha = write_m03r_v11_initial_parameter_state(path, _Policy())
    original = path.read_bytes()
    path.chmod(0o640)
    path.write_bytes(original + b"tamper")
    with pytest.raises(M03RV11InitialStateError, match="file hash"):
        load_m03r_v11_initial_parameter_state(
            path,
            _Policy(),
            expected_file_sha256=file_sha,
            expected_state_sha256=state_sha,
        )

    clean = tmp_path / "clean.pt"
    state_sha, file_sha = write_m03r_v11_initial_parameter_state(clean, _Policy())
    incompatible = nn.Linear(2, 2)
    with pytest.raises(M03RV11InitialStateError, match="strictly match"):
        load_m03r_v11_initial_parameter_state(
            clean,
            incompatible,
            expected_file_sha256=file_sha,
            expected_state_sha256=state_sha,
        )
