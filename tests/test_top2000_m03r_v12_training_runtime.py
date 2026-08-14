from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v12_top2000_dev import (
    M03R_V12_SETTING_IDS,
)
from rl_quant.training.top2000_m03r_v7_dev import (
    render_top2000_m03r_v7_development_folds,
)
from rl_quant.training.top2000_m03r_v10_fold import render_m03r_v10_fold_geometry
from rl_quant.training.top2000_m03r_v12_predictive_worker import (
    M03RV12PredictiveWorkerPlan,
)
from rl_quant.training.top2000_m03r_v12_schedule import (
    M03RV12PanelEpisodeSchedule,
)
from rl_quant.training.top2000_m03r_v12_training_runtime import (
    M03RV12TrainingRuntimeError,
    run_m03r_v12_pretraining_fold_update,
)


class _Validated:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)

    def validate(self) -> None:
        return None

    def validate_unmodified(self) -> None:
        return None


def _schedule_and_worker() -> tuple[
    M03RV12PanelEpisodeSchedule,
    M03RV12PredictiveWorkerPlan,
]:
    folds = render_top2000_m03r_v7_development_folds(1001)
    schedule = M03RV12PanelEpisodeSchedule(
        protocol_common_data_sha256="a" * 64,
        cache_sha256="b" * 64,
        fold_geometry_sha256=tuple(
            render_m03r_v10_fold_geometry(fold).receipt_sha256 for fold in folds
        ),
    )
    worker = M03RV12PredictiveWorkerPlan(
        setting_index=1,
        setting_id=M03R_V12_SETTING_IDS[1],
        output_root="/approved/v12/setting-1",
        cache_path="/approved/cache.pt",
        initial_parameter_state_path="/approved/common-initial-state.pt",
        panel_episode_schedule_sha256=schedule.receipt_sha256,
        initial_parameter_state_file_sha256="9" * 64,
        initial_parameter_state_sha256="c" * 64,
        cache_sha256="b" * 64,
        risk_source_manifest_path="/approved/risk.json",
        risk_source_manifest_file_sha256="d" * 64,
        projector_manifest_path="/approved/projector.json",
        projector_manifest_file_sha256="2" * 64,
        projector_manifest_sha256="3" * 64,
        projector_binding_sha256="e" * 64,
        source_manifest_sha256="f" * 64,
        source_archive_sha256="1" * 64,
        structural_preflight_path="/approved/preflight.json",
        structural_preflight_file_sha256="4" * 64,
        structural_preflight_receipt_sha256="5" * 64,
    )
    return schedule, worker


def test_v12_training_runtime_binds_one_exact_paired_rank_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_quant.training.top2000_m03r_v12_training_runtime as runtime

    schedule, worker = _schedule_and_worker()
    fold = render_top2000_m03r_v7_development_folds(1001)[2]
    cache = _Validated(cache_sha256="b" * 64, action_hash="2" * 64)
    risk = _Validated(
        cache_sha256="b" * 64,
        action_hash="2" * 64,
        exposures=object(),
    )
    written = SimpleNamespace(manifest_file_sha256="d" * 64)
    sequence = SimpleNamespace(decision_state=torch.ones((3, 2, 4, 1)))
    inputs = SimpleNamespace(
        daily_ohlcv=torch.arange(24, dtype=torch.float32).reshape(3, 2, 4),
        availability=torch.ones((3, 2), dtype=torch.bool),
        past_returns=torch.zeros((3, 2, 4)),
    )
    built = SimpleNamespace(
        sequence=object(),
        identity=SimpleNamespace(receipt_sha256="3" * 64),
    )
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        runtime,
        "build_top2000_hold30_development_sequence_from_loaded_cache",
        lambda *args, **kwargs: built,
    )
    monkeypatch.setattr(
        runtime, "_move_and_bind_sequence", lambda *args, **kwargs: sequence
    )
    monkeypatch.setattr(
        runtime, "top2000_m03r_v7_decision_inputs", lambda value: inputs
    )
    monkeypatch.setattr(
        runtime,
        "replace",
        lambda value, **kwargs: SimpleNamespace(
            decision_state=kwargs["decision_state"]
        ),
    )

    class _Provider:
        def __init__(self, value: object) -> None:
            assert value is inputs

        def replay_origin_states(
            self,
            source_policy: object,
            bound_sequence: object,
            local_origins: torch.Tensor,
        ) -> torch.Tensor:
            captured["local_origins"] = tuple(int(row) for row in local_origins)
            return torch.ones((local_origins.numel(), 2, 4))

    monkeypatch.setattr(runtime, "Top2000M03RV7DecisionStateProvider", _Provider)
    batch = object()
    monkeypatch.setattr(
        runtime,
        "build_m03r_v12_batch_from_origin_states",
        lambda *args, **kwargs: batch,
    )

    def _step(
        policy: object,
        observed_batch: object,
        optimizer: object,
        partition: object,
        shard: object,
        paired: object,
        **kwargs: object,
    ) -> _Validated:
        assert observed_batch is batch
        captured["shard"] = shard
        captured["paired"] = paired
        return _Validated(
            training_shard_receipt_sha256=shard.receipt_sha256,
            paired_input_receipt_sha256=paired.receipt_sha256,
        )

    monkeypatch.setattr(runtime, "train_m03r_v12_predictive_batch_update", _step)
    policy = SimpleNamespace(source_policy=object())
    result = run_m03r_v12_pretraining_fold_update(
        cache,
        worker,
        schedule,
        fold,
        risk,
        written,
        policy,
        object(),
        object(),
        completed_updates=1,
        distributed_rank=1,
        distributed_world_size=2,
        device=torch.device("cpu"),
    )
    result.validate()
    expected_global = result.training_shard.rank_origins[1]
    assert captured["local_origins"] == tuple(
        value - result.training_shard.episode_start for value in expected_global
    )
    assert result.paired_input.input_tensor_sha256
    assert captured["paired"].receipt_sha256 == result.paired_input.receipt_sha256


def test_v12_training_runtime_rejects_wrong_external_risk_manifest_before_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_quant.training.top2000_m03r_v12_training_runtime as runtime

    schedule, worker = _schedule_and_worker()
    fold = render_top2000_m03r_v7_development_folds(1001)[0]
    cache = _Validated(cache_sha256="b" * 64, action_hash="2" * 64)
    risk = _Validated(
        cache_sha256="b" * 64,
        action_hash="2" * 64,
        exposures=object(),
    )
    called = False

    def _unexpected(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("episode build must not start")

    monkeypatch.setattr(
        runtime,
        "build_top2000_hold30_development_sequence_from_loaded_cache",
        _unexpected,
    )
    with pytest.raises(M03RV12TrainingRuntimeError, match="risk"):
        run_m03r_v12_pretraining_fold_update(
            cache,
            worker,
            schedule,
            fold,
            risk,
            SimpleNamespace(manifest_file_sha256="9" * 64),
            SimpleNamespace(source_policy=object()),
            object(),
            object(),
            completed_updates=1,
            distributed_rank=0,
            distributed_world_size=2,
            device=torch.device("cpu"),
        )
    assert not called
