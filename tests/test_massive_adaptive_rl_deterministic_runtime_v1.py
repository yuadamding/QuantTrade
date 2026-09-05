from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from rl_quant.workflows.massive_adaptive_rl_deterministic_runtime_v1 import (
    MASSIVE_ADAPTIVE_RL_DETERMINISTIC_RUNTIME_V1_ENVIRONMENT,
    massive_adaptive_rl_deterministic_environment_v1,
)
from rl_quant.workflows.massive_adaptive_rl_vertical_qualification_scope_v1 import (
    massive_adaptive_rl_vertical_qualification_experiment_v1,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_deterministic_environment_overrides_ambient_thread_settings() -> None:
    environment = massive_adaptive_rl_deterministic_environment_v1(
        {
            "PATH": os.environ.get("PATH", ""),
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "4",
            "PYTHONHASHSEED": "99",
        }
    )
    assert {
        name: environment[name]
        for name, _ in MASSIVE_ADAPTIVE_RL_DETERMINISTIC_RUNTIME_V1_ENVIRONMENT
    } == dict(MASSIVE_ADAPTIVE_RL_DETERMINISTIC_RUNTIME_V1_ENVIRONMENT)
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_qualification_namespace_is_permanently_nonproduction() -> None:
    assert massive_adaptive_rl_vertical_qualification_experiment_v1(
        "v5-vertical-qualification-smoke"
    )
    assert not massive_adaptive_rl_vertical_qualification_experiment_v1(
        "production-experiment"
    )


def test_cli_launcher_establishes_runtime_in_a_fresh_process() -> None:
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "6",
            "NUMEXPR_NUM_THREADS": "3",
            "PYTHONHASHSEED": "91",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    environment.pop("QUANTTRADE_ADAPTIVE_RL_RUNTIME_V1", None)
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "rl_quant.workflows.massive_adaptive_rl_cli_v1",
            "--help",
        ),
        cwd=_repository_root(),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


def test_qualification_runner_configures_torch_before_pytest() -> None:
    environment = massive_adaptive_rl_deterministic_environment_v1(os.environ)
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "rl_quant.workflows.massive_adaptive_rl_vertical_qualification_runner_v1",
            "--version",
        ),
        cwd=_repository_root(),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "pytest" in completed.stdout.lower()
