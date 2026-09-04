from __future__ import annotations

from pathlib import Path

import pytest

from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    MassiveAdaptiveRLExperimentLockV1Unavailable,
    massive_adaptive_rl_experiment_lock_relative_path_v1,
    massive_adaptive_rl_experiment_orchestration_lock_v1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v3 import (
    _experiment_v3_orchestration_lease,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v2 import (
    _massive_adaptive_rl_experiment_orchestration_lease_v1,
)
from rl_quant.workflows.massive_adaptive_rl_experiment_runner_v4 import (
    MassiveAdaptiveRLExperimentRunnerV4LeaseUnavailable,
    _experiment_v4_orchestration_lease,
)


def test_v2_through_v5_use_one_physical_experiment_lock(tmp_path: Path) -> None:
    experiment_id = "shared-lock"
    relative = massive_adaptive_rl_experiment_lock_relative_path_v1(
        experiment_id=experiment_id
    )
    with _experiment_v3_orchestration_lease(
        artifact_root=tmp_path, experiment_id=experiment_id
    ):
        assert (tmp_path / relative).is_file()
        with pytest.raises(MassiveAdaptiveRLExperimentLockV1Unavailable):
            with massive_adaptive_rl_experiment_orchestration_lock_v1(
                artifact_root=tmp_path, experiment_id=experiment_id
            ):
                pass

    with _massive_adaptive_rl_experiment_orchestration_lease_v1(
        artifact_root=tmp_path, experiment_id=experiment_id
    ):
        with pytest.raises(MassiveAdaptiveRLExperimentLockV1Unavailable):
            with massive_adaptive_rl_experiment_orchestration_lock_v1(
                artifact_root=tmp_path, experiment_id=experiment_id
            ):
                pass

    with massive_adaptive_rl_experiment_orchestration_lock_v1(
        artifact_root=tmp_path, experiment_id=experiment_id
    ):
        with pytest.raises(MassiveAdaptiveRLExperimentRunnerV4LeaseUnavailable):
            with _experiment_v4_orchestration_lease(
                artifact_root=tmp_path, experiment_id=experiment_id
            ):
                pass


def test_global_lock_preserves_body_oserror(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="body failed"):
        with massive_adaptive_rl_experiment_orchestration_lock_v1(
            artifact_root=tmp_path,
            experiment_id="body-error",
        ):
            raise OSError("body failed")
