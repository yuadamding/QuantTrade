from __future__ import annotations

from dataclasses import replace

import pytest

from rl_quant.protocol.hold30_alpha_m03r_v11_top2000_dev import (
    M03R_V11_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v11_predictive_worker import (
    M03RV11PredictivePanelPlan,
    M03RV11PredictiveWorkerError,
    M03RV11PredictiveWorkerPlan,
)


def _worker(setting: int) -> M03RV11PredictiveWorkerPlan:
    return M03RV11PredictiveWorkerPlan(
        setting_index=setting,
        setting_id=M03R_V11_SETTING_IDS[setting],
        output_root=f"/approved/v11/setting-{setting}",
        cache_path="/approved/cache.pt",
        initial_parameter_state_path="/approved/common-initial-state.pt",
        panel_episode_schedule_sha256="a" * 64,
        initial_parameter_state_file_sha256="9" * 64,
        initial_parameter_state_sha256="b" * 64,
        cache_sha256="c" * 64,
        risk_source_manifest_path="/approved/risk.json",
        risk_source_manifest_file_sha256="d" * 64,
        projector_manifest_path="/approved/projector.json",
        projector_manifest_file_sha256="2" * 64,
        projector_manifest_sha256="3" * 64,
        projector_binding_sha256="e" * 64,
        source_manifest_sha256="f" * 64,
        source_archive_sha256="1" * 64,
    )


def test_v11_panel_requires_shared_schedule_inputs_and_initial_state() -> None:
    panel = M03RV11PredictivePanelPlan(tuple(_worker(index) for index in range(3)))
    panel.validate()
    assert panel.maximum_h100_requests == 6
    assert not panel.package_authorized
    assert not panel.kubernetes_launch_authorized
    assert not panel.economic_panel_authorized
    assert len({row.panel_episode_schedule_sha256 for row in panel.workers}) == 1
    assert len({row.initial_parameter_state_sha256 for row in panel.workers}) == 1


def test_v11_panel_rejects_setting_specific_schedule_or_launch_authority() -> None:
    workers = tuple(_worker(index) for index in range(3))
    drifted = (
        workers[0],
        replace(workers[1], panel_episode_schedule_sha256="2" * 64),
        workers[2],
    )
    with pytest.raises(M03RV11PredictiveWorkerError, match="panel plan"):
        M03RV11PredictivePanelPlan(drifted).validate()
    with pytest.raises(M03RV11PredictiveWorkerError, match="worker plan"):
        replace(workers[0], kubernetes_launch_authorized=True).validate()
