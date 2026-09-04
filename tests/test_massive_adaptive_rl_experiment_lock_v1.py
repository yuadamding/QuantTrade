from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Any

import pytest

from rl_quant.workflows.massive_adaptive_rl_experiment_lock_v1 import (
    MassiveAdaptiveRLExperimentLockV1Unavailable,
    massive_adaptive_rl_artifact_root_writer_lock_v1,
    massive_adaptive_rl_experiment_lock_relative_path_v1,
    massive_adaptive_rl_experiment_materialization_lock_v1,
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


def _attempt_experiment_lock(
    root: str,
    experiment_id: str,
    result_queue: Any,
) -> None:
    try:
        with massive_adaptive_rl_experiment_orchestration_lock_v1(
            artifact_root=root,
            experiment_id=experiment_id,
        ):
            result_queue.put("acquired")
    except MassiveAdaptiveRLExperimentLockV1Unavailable:
        result_queue.put("unavailable")


def _attempt_artifact_root_lock(root: str, result_queue: Any) -> None:
    try:
        with massive_adaptive_rl_artifact_root_writer_lock_v1(
            artifact_root=root,
        ):
            result_queue.put("acquired")
    except MassiveAdaptiveRLExperimentLockV1Unavailable:
        result_queue.put("unavailable")


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


def test_experiment_lock_excludes_a_separate_process(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    with massive_adaptive_rl_experiment_orchestration_lock_v1(
        artifact_root=tmp_path,
        experiment_id="cross-process",
    ):
        process = context.Process(
            target=_attempt_experiment_lock,
            args=(str(tmp_path), "cross-process", results),
        )
        process.start()
        process.join(timeout=15)
        assert process.exitcode == 0
        assert results.get(timeout=2) == "unavailable"
    results.close()
    results.join_thread()


def test_materializer_reuses_the_current_context_experiment_lock(
    tmp_path: Path,
) -> None:
    with massive_adaptive_rl_experiment_orchestration_lock_v1(
        artifact_root=tmp_path,
        experiment_id="nested-materializer",
    ):
        with massive_adaptive_rl_experiment_materialization_lock_v1(
            artifact_root=tmp_path,
            experiment_id="nested-materializer",
        ):
            pass


def test_artifact_root_writer_lock_excludes_a_separate_process(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    with massive_adaptive_rl_artifact_root_writer_lock_v1(
        artifact_root=tmp_path,
    ):
        process = context.Process(
            target=_attempt_artifact_root_lock,
            args=(str(tmp_path), results),
        )
        process.start()
        process.join(timeout=15)
        assert process.exitcode == 0
        assert results.get(timeout=2) == "unavailable"
    results.close()
    results.join_thread()
