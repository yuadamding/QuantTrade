from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v15_top2000_dev import M03R_V15_SETTING_IDS
from rl_quant.training.top2000_m03r_v15_predictive_worker import (
    M03R_V15_FOLD_UPDATE_COUNTS,
    M03RV15PredictivePanelPlan,
    M03RV15PredictiveWorkerError,
    M03RV15PredictiveWorkerPlan,
)


def _worker(index: int) -> M03RV15PredictiveWorkerPlan:
    return M03RV15PredictiveWorkerPlan(
        setting_index=index,
        setting_id=M03R_V15_SETTING_IDS[index],
        output_root=f"/approved/v15/setting-{index}",
        cache_path="/approved/cache.pt",
        initial_parameter_state_path="/approved/common-initial-state.pt",
        panel_episode_schedule_sha256="1" * 64,
        initial_parameter_state_file_sha256="2" * 64,
        initial_parameter_state_sha256="3" * 64,
        initial_parameter_architecture_sha256="c" * 64,
        cache_sha256="4" * 64,
        risk_source_manifest_path="/approved/risk.json",
        risk_source_manifest_file_sha256="5" * 64,
        projector_manifest_path="/approved/projector.json",
        projector_manifest_file_sha256="d" * 64,
        projector_manifest_sha256="e" * 64,
        projector_binding_sha256="f" * 64,
        source_manifest_sha256="6" * 64,
        source_archive_sha256="7" * 64,
        structural_preflight_path="/approved/preflight.json",
        structural_preflight_file_sha256="8" * 64,
        structural_preflight_receipt_sha256="9" * 64,
    )


def test_v15_worker_binds_variable_equal_epoch_fold_updates() -> None:
    worker = _worker(0)
    worker.validate()
    assert M03R_V15_FOLD_UPDATE_COUNTS == (24, 32, 48, 56, 72, 80)
    assert worker.fold_optimizer_updates == M03R_V15_FOLD_UPDATE_COUNTS
    assert worker.economic_optimizer_updates == 0
    assert worker.outer_2026_access_authorized is False
    assert worker.package_authorized is False
    assert worker.kubernetes_launch_authorized is False


def test_v15_panel_is_two_paired_workers_and_four_h100_requests() -> None:
    panel = M03RV15PredictivePanelPlan((_worker(0), _worker(1)))
    panel.validate()
    assert panel.indexed_completions == 2
    assert panel.parallelism == 2
    assert panel.maximum_h100_requests == 4
    assert panel.economic_panel_authorized is False


def test_v15_worker_rejects_uniform_64_update_reversion() -> None:
    with pytest.raises(M03RV15PredictiveWorkerError, match="worker plan drifted"):
        replace(_worker(0), fold_optimizer_updates=(64,) * 6).validate()
