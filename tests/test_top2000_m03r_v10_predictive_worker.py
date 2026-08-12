from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v10_top2000_dev import (
    M03R_V10_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v10_predictive_worker import (
    M03RV10PredictivePanelPlan,
    M03RV10PredictiveWorkerError,
    M03RV10PredictiveWorkerPlan,
    resolve_m03r_v10_predictive_setting_index,
)


def _worker(setting: int) -> M03RV10PredictiveWorkerPlan:
    return M03RV10PredictiveWorkerPlan(
        setting_index=setting,
        setting_id=M03R_V10_SETTING_IDS[setting],
        cache_path="/approved/cache.pt",
        cache_sha256="a" * 64,
        risk_source_manifest_path="/approved/risk-source.json",
        risk_source_manifest_file_sha256="b" * 64,
        projector_manifest_path="/approved/projector.json",
        projector_manifest_file_sha256="c" * 64,
        projector_manifest_sha256="d" * 64,
        projector_binding_sha256="e" * 64,
        source_manifest_sha256="f" * 64,
        source_archive_sha256=f"{setting + 1:x}" * 64,
        output_root=f"/approved/output/setting-{setting}",
    )


def test_v10_panel_is_three_workers_six_h100_predictive_only() -> None:
    panel = M03RV10PredictivePanelPlan(tuple(_worker(index) for index in range(3)))
    panel.validate()
    assert panel.maximum_h100_requests == 6
    assert not panel.economic_panel_authorized
    assert all(not worker.v9_state_reuse_authorized for worker in panel.workers)
    assert all(not worker.outer_2026_access_authorized for worker in panel.workers)


def test_v10_completion_mapping_and_resource_drift_fail_closed() -> None:
    assert tuple(
        resolve_m03r_v10_predictive_setting_index(index) for index in range(3)
    ) == (0, 1, 2)
    for value in (-1, 3, True):
        with pytest.raises(M03RV10PredictiveWorkerError, match="JOB_COMPLETION_INDEX"):
            resolve_m03r_v10_predictive_setting_index(value)  # type: ignore[arg-type]
    workers = tuple(_worker(index) for index in range(3))
    with pytest.raises(M03RV10PredictiveWorkerError, match="panel plan"):
        replace(
            M03RV10PredictivePanelPlan(workers), maximum_h100_requests=16
        ).validate()
    with pytest.raises(M03RV10PredictiveWorkerError, match="worker plan"):
        replace(workers[0], economic_optimizer_updates=64).validate()
