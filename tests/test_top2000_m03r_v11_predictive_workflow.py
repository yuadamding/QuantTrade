from __future__ import annotations

import inspect

import pytest

from rl_quant.workflows.top2000_m03r_v11_predictive import (
    M03RV11PredictiveWorkflowError,
    resolve_m03r_v11_completion_index,
    run_m03r_v11_predictive_worker,
)


def test_v11_completion_index_is_exact_and_never_offset() -> None:
    assert tuple(resolve_m03r_v11_completion_index(value) for value in range(3)) == (
        0,
        1,
        2,
    )
    with pytest.raises(M03RV11PredictiveWorkflowError, match="0, 1, or 2"):
        resolve_m03r_v11_completion_index(3)
    with pytest.raises(M03RV11PredictiveWorkflowError, match="0, 1, or 2"):
        resolve_m03r_v11_completion_index(True)


def test_v11_worker_routes_only_corrected_training_checkpoint_and_gate() -> None:
    source = inspect.getsource(run_m03r_v11_predictive_worker)
    assert "run_m03r_v11_pretraining_fold_update" in source
    assert "write_immutable_m03r_v11_alpha_checkpoint" in source
    assert "load_m03r_v11_alpha_checkpoint_for_evaluation" in source
    assert "evaluate_m03r_v11_loaded_qualification_fold" in source
    assert "qualify_m03r_v11_predictive_candidate" in source
    assert "build_m03r_v11_bootstrap_plan" in source
    assert "load_m03r_v11_initial_parameter_state" in source
    assert 'economic_optimizer_updates": 0' in source
    assert 'outer_2026_accessed": False' in source
    assert "run_m03r_v9_pretraining_fold_update" not in source
    assert "evaluate_m03r_v9_qualification_fold" not in source


def test_v11_capacity_terminal_binds_setting_and_nccl_startup_contract() -> None:
    source = inspect.getsource(run_m03r_v11_predictive_worker)
    capacity = source.split("if startup_only:", 1)[1].split("cache =", 1)[0]
    assert '"setting_index": worker.setting_index' in capacity
    assert '"setting_id": worker.setting_id' in capacity
    assert '"nccl_process_group_initialized": True' in capacity
    assert "load_m03r_v11_initial_parameter_state" in capacity
    assert '"packaged_initial_state_loaded": True' in capacity


def test_v11_capacity_probe_requires_disjoint_output_before_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_quant.workflows.top2000_m03r_v11_predictive as workflow

    package = object()
    authorization = type(
        "Authorization",
        (),
        {"package_plan_file_sha256": "a" * 64},
    )()
    monkeypatch.setattr(workflow, "load_m03r_v11_package_plan", lambda *a, **k: package)
    monkeypatch.setattr(
        workflow,
        "load_m03r_v11_execution_authorization",
        lambda *a, **k: authorization,
    )
    monkeypatch.setattr(
        workflow,
        "_resolve_worker",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must reject first")),
    )
    with pytest.raises(M03RV11PredictiveWorkflowError, match="disjoint"):
        run_m03r_v11_predictive_worker(
            "/package.json",
            "/authorization.json",
            expected_package_plan_file_sha256="a" * 64,
            expected_authorization_file_sha256="b" * 64,
            completion_index=0,
            startup_only=True,
        )
