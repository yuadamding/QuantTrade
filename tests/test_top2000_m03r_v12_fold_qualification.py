from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.training.top2000_m03r_v12_checkpoint import M03RV12LoadedCheckpoint
from rl_quant.training.top2000_m03r_v12_policy import M03RV12HeadIdentity
from rl_quant.training.top2000_m03r_v12_fold_qualification import (
    M03RV12FoldQualificationError,
    evaluate_m03r_v12_loaded_qualification_fold,
)
from rl_quant.training.top2000_m03r_v12_predictive_worker import (
    M03RV12PredictiveWorkerPlan,
)


class _Validated:
    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)

    def validate(self) -> None:
        return None

    def validate_unmodified(self) -> None:
        return None


def _worker() -> M03RV12PredictiveWorkerPlan:
    return M03RV12PredictiveWorkerPlan(
        setting_index=1,
        setting_id=M03R_V12_SETTING_IDS[1],
        output_root="/approved/v12/setting-1",
        cache_path="/approved/cache.pt",
        initial_parameter_state_path="/approved/common-initial-state.pt",
        panel_episode_schedule_sha256="a" * 64,
        initial_parameter_state_file_sha256="9" * 64,
        initial_parameter_state_sha256="b" * 64,
        cache_sha256="c" * 64,
        risk_source_manifest_path="/approved/risk.json",
        risk_source_manifest_file_sha256="d" * 64,
        projector_manifest_path="/approved/projector.json",
        projector_manifest_file_sha256="8" * 64,
        projector_manifest_sha256="9" * 64,
        projector_binding_sha256="e" * 64,
        source_manifest_sha256="f" * 64,
        source_archive_sha256="1" * 64,
        structural_preflight_path="/approved/preflight.json",
        structural_preflight_file_sha256="4" * 64,
        structural_preflight_receipt_sha256="5" * 64,
    )


def test_v12_fold_qualification_routes_only_corrected_batch_and_sleeve() -> None:
    source = inspect.getsource(evaluate_m03r_v12_loaded_qualification_fold)
    assert "build_m03r_v12_batch_from_origin_states" in source
    assert "run_m03r_v12_simple_sleeve" in source
    assert "build_m03r_v12_fold_qualification_lineage" in source
    assert "source_receipt_sha256=built.identity.receipt_sha256" in source
    assert "source_receipt_sha256=loaded.source_array_sha256" not in source
    assert "run_m03r_v9_simple_sleeve" not in source
    assert "project_m03r_v9_signal_to_exposure_null" not in source


def test_v12_fold_qualification_rejects_checkpoint_setting_before_episode_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_quant.training.top2000_m03r_v12_fold_qualification as qualification

    called = False

    def _unexpected(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("qualification episode must not be built")

    monkeypatch.setattr(
        qualification,
        "build_top2000_hold30_development_sequence_from_loaded_cache",
        _unexpected,
    )
    loaded = M03RV12LoadedCheckpoint(
        setting_index=0,
        setting_id=M03R_V12_SETTING_IDS[0],
        fold_index=0,
        completed_updates=64,
        selected_horizon_sessions=3,
        model_state_sha256="2" * 64,
        checkpoint_file_sha256="3" * 64,
        episode_schedule_sha256="a" * 64,
        residual_operator_root_sha256="4" * 64,
        source_array_sha256="5" * 64,
        asset_axis_sha256="6" * 64,
        head_identity=M03RV12HeadIdentity(
            setting_id=M03R_V12_SETTING_IDS[0],
            selected_alpha_horizon=3,
            economic_mean_head_state_sha256="7" * 64,
            economic_scale_head_state_sha256="8" * 64,
            rank_score_head_state_sha256="9" * 64,
        ),
    )
    with pytest.raises(M03RV12FoldQualificationError, match="drifted"):
        evaluate_m03r_v12_loaded_qualification_fold(
            _Validated(cache_sha256="c" * 64, action_hash="6" * 64),
            _worker(),
            render_top2000_m03r_v7_development_folds(1001)[0],
            _Validated(exposures=SimpleNamespace(receipt_sha256="7" * 64)),
            _Validated(),
            SimpleNamespace(),
            loaded,
            device=torch.device("cpu"),
        )
    assert not called
