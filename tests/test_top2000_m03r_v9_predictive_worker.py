from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v9_top2000_dev import (
    M03RV9HorizonBinding,
)
from rl_quant.training.top2000_m03r_v9_economic_worker import (
    M03RV9EconomicWorkerNotAuthorized,
    require_m03r_v9_economic_generation,
)
from rl_quant.training.top2000_m03r_v9_predictive_worker import (
    M03RV9PredictivePanelPlan,
    M03RV9PredictiveWorkerError,
    M03RV9PredictiveWorkerPlan,
    resolve_m03r_v9_predictive_setting_index,
)
from rl_quant.training.top2000_m03r_v9_selection import (
    M03RV9PredictiveQualification,
)
from rl_quant.workflows.top2000_m03r_v9_predictive import (
    M03RV9PredictiveWorkflowError,
    resolve_m03r_v9_completion_index,
)


def _worker(setting: int) -> M03RV9PredictiveWorkerPlan:
    return M03RV9PredictiveWorkerPlan(
        setting_index=setting,
        setting_id=(
            "V9-P0-factor-residual-ranked",
            "V9-P1-factor-residual-no-ranking",
            "V9-P2-benchmark-relative-ranked",
        )[setting],
        cache_path="/approved/cache.pt",
        cache_sha256="a" * 64,
        risk_source_manifest_path="/approved/risk-source-manifest.json",
        risk_source_manifest_file_sha256="b" * 64,
        projector_manifest_path="/approved/projector-manifest.json",
        projector_manifest_file_sha256="c" * 64,
        projector_manifest_sha256="d" * 64,
        projector_binding_sha256="e" * 64,
        source_manifest_sha256="f" * 64,
        source_archive_sha256=f"{setting + 1:x}" * 64,
        output_root=f"/approved/output/setting-{setting}",
    )


def test_panel_is_exactly_three_workers_six_h100_and_zero_economic_updates() -> None:
    panel = M03RV9PredictivePanelPlan(tuple(_worker(index) for index in range(3)))
    panel.validate()
    assert panel.indexed_completions == 3
    assert panel.maximum_h100_requests == 6
    assert not panel.economic_panel_authorized
    assert all(worker.economic_optimizer_updates == 0 for worker in panel.workers)
    assert all(not worker.early_stopping_enabled for worker in panel.workers)
    assert all(
        worker.qualification_evaluation_updates == (64,) for worker in panel.workers
    )


def test_completion_index_mapping_is_direct_and_fail_closed() -> None:
    assert tuple(
        resolve_m03r_v9_predictive_setting_index(index) for index in range(3)
    ) == (0, 1, 2)
    for value in (-1, 3, True):
        with pytest.raises(M03RV9PredictiveWorkerError, match="JOB_COMPLETION_INDEX"):
            resolve_m03r_v9_predictive_setting_index(value)  # type: ignore[arg-type]


def test_panel_rejects_resource_or_economic_drift() -> None:
    workers = tuple(_worker(index) for index in range(3))
    with pytest.raises(M03RV9PredictiveWorkerError, match="panel plan drifted"):
        replace(M03RV9PredictivePanelPlan(workers), maximum_h100_requests=16).validate()
    with pytest.raises(M03RV9PredictiveWorkerError, match="worker plan drifted"):
        replace(workers[0], economic_optimizer_updates=64).validate()


def test_even_a_passing_predictive_receipt_cannot_run_economic_training() -> None:
    binding = M03RV9HorizonBinding(30, 30, 30)
    qualification = M03RV9PredictiveQualification(
        setting_id="V9-P0-factor-residual-ranked",
        selected_horizon_sessions=30,
        horizon_binding_sha256=binding.receipt_sha256,
        fold_alpha_receipt_sha256=tuple(f"{index + 1:x}" * 64 for index in range(6)),
        fold_sleeve_receipt_sha256=tuple(f"{index + 7:x}" * 64 for index in range(6)),
        mean_rank_ic=0.03,
        positive_rank_ic_fold_count=6,
        mean_top_bottom_spread=0.001,
        positive_spread_fold_count=6,
        mean_simple_sleeve_gross_active_return=0.01,
        mean_simple_sleeve_net_active_return_10bp=0.005,
        mean_simple_sleeve_net_active_return_10bp_lcb=0.001,
        gross_positive_fold_count=6,
        mean_break_even_one_way_cost_basis_points=20.0,
        passed=True,
        economic_generation_may_be_minted=True,
        economic_panel_authorized=False,
    )
    with pytest.raises(
        M03RV9EconomicWorkerNotAuthorized, match="new source-homogeneous"
    ):
        require_m03r_v9_economic_generation(qualification)


def test_workflow_uses_only_the_exact_three_row_index_mapping(monkeypatch) -> None:
    monkeypatch.setenv("JOB_COMPLETION_INDEX", "2")
    assert resolve_m03r_v9_completion_index(None) == 2
    with pytest.raises(M03RV9PredictiveWorkerError, match="JOB_COMPLETION_INDEX"):
        resolve_m03r_v9_completion_index(3)
    monkeypatch.setenv("JOB_COMPLETION_INDEX", "economic")
    with pytest.raises(M03RV9PredictiveWorkflowError, match="integer"):
        resolve_m03r_v9_completion_index(None)
