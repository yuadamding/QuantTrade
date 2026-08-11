from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v8_top2000_dev import (
    M03R_V8_TOP2000_DEV_SETTINGS,
)
from rl_quant.training.top2000_m03r_v8_plan import (
    M03RV8DevelopmentTrainingPlan,
)
from rl_quant.workflows.top2000_m03r_v8_pretraining import (
    M03RV8PretrainingWorkerError,
    resolve_m03r_v8_pretraining_plan_path,
    run_pretraining_worker,
)


def _write_plan(tmp_path, setting_index: int):
    plan = M03RV8DevelopmentTrainingPlan(
        setting_index=setting_index,
        setting_id=M03R_V8_TOP2000_DEV_SETTINGS[setting_index].setting_id,
        cache_path="/immutable/cache.pt",
        cache_sha256="a" * 64,
        output_root=str(tmp_path / "output"),
        source_manifest_sha256="b" * 64,
    )
    path = tmp_path / f"plan-{setting_index}.json"
    path.write_text(json.dumps(asdict(plan)))
    return path


def test_no_pretraining_control_is_rejected_before_gpu_startup(tmp_path) -> None:
    with pytest.raises(M03RV8PretrainingWorkerError, match="no predictive"):
        run_pretraining_worker(_write_plan(tmp_path, 1))


def test_qualification_update_count_is_exact(tmp_path) -> None:
    with pytest.raises(M03RV8PretrainingWorkerError, match="exactly four"):
        run_pretraining_worker(
            _write_plan(tmp_path, 0),
            qualification_updates=3,
        )


def test_indexed_completion_mapping_excludes_no_pretraining_control(
    tmp_path, monkeypatch
) -> None:
    directory = str(tmp_path / "plans")
    assert resolve_m03r_v8_pretraining_plan_path(
        plan=None,
        plan_directory=directory,
        completion_index=0,
    ).name == "setting-00.json"
    assert resolve_m03r_v8_pretraining_plan_path(
        plan=None,
        plan_directory=directory,
        completion_index=6,
    ).name == "setting-07.json"
    monkeypatch.setenv("JOB_COMPLETION_INDEX", "1")
    assert resolve_m03r_v8_pretraining_plan_path(
        plan=None,
        plan_directory=directory,
        completion_index=None,
    ).name == "setting-02.json"
    with pytest.raises(M03RV8PretrainingWorkerError, match=r"\[0, 6\]"):
        resolve_m03r_v8_pretraining_plan_path(
            plan=None,
            plan_directory=directory,
            completion_index=7,
        )
