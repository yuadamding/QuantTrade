from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from rl_quant.protocol.hold30_alpha_m03r_v13_top2000_dev import M03R_V13_SETTING_IDS
from rl_quant.training.top2000_m03r_v13_fold import (
    M03RV13PanelEpisodeSchedule,
    render_m03r_v13_fold_geometries,
)
from rl_quant.training.top2000_m03r_v13_predictive_worker import (
    M03RV13PredictiveWorkerPlan,
)
from rl_quant.training.top2000_m03r_v13_training_runtime import (
    M03RV13TrainingRuntimeError,
    run_m03r_v13_pretraining_fold_update,
)


class _Validated:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)

    def validate(self) -> None:
        return None

    def validate_unmodified(self) -> None:
        return None


def _schedule_and_worker() -> tuple[
    M03RV13PanelEpisodeSchedule,
    M03RV13PredictiveWorkerPlan,
]:
    geometries = render_m03r_v13_fold_geometries(1001)
    schedule = M03RV13PanelEpisodeSchedule(
        protocol_common_data_sha256="a" * 64,
        cache_sha256="b" * 64,
        asset_axis_sha256="2" * 64,
        fold_geometry_sha256=tuple(row.receipt_sha256 for row in geometries),
    )
    worker = M03RV13PredictiveWorkerPlan(
        setting_index=0,
        setting_id=M03R_V13_SETTING_IDS[0],
        output_root="/approved/v13/setting-0",
        cache_path="/approved/cache.pt",
        initial_parameter_state_path="/approved/common-initial-state.pt",
        panel_episode_schedule_sha256=schedule.receipt_sha256,
        initial_parameter_state_file_sha256="9" * 64,
        initial_parameter_state_sha256="c" * 64,
        initial_parameter_architecture_sha256="8" * 64,
        cache_sha256="b" * 64,
        risk_source_manifest_path="/approved/risk.json",
        risk_source_manifest_file_sha256="d" * 64,
        projector_manifest_path="/approved/projector.json",
        projector_manifest_file_sha256="6" * 64,
        projector_manifest_sha256="7" * 64,
        projector_binding_sha256="e" * 64,
        source_manifest_sha256="f" * 64,
        source_archive_sha256="1" * 64,
        structural_preflight_path="/approved/preflight.json",
        structural_preflight_file_sha256="4" * 64,
        structural_preflight_receipt_sha256="5" * 64,
    )
    return schedule, worker


def test_v13_runtime_uses_exact_full_context_rank_shard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_quant.training.top2000_m03r_v13_training_runtime as runtime

    schedule, worker = _schedule_and_worker()
    geometry = render_m03r_v13_fold_geometries(1001)[2]
    cache = _Validated(cache_sha256="b" * 64, action_hash="2" * 64)
    risk = _Validated(
        cache_sha256="b" * 64,
        action_hash="2" * 64,
        exposures=object(),
    )
    written = SimpleNamespace(manifest_file_sha256="d" * 64)
    sequence = SimpleNamespace(decision_state=torch.ones((378, 1, 4, 1)))
    built = SimpleNamespace(
        sequence=object(),
        identity=SimpleNamespace(receipt_sha256="3" * 64),
    )
    inputs = object()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        runtime,
        "build_top2000_hold30_development_sequence_from_loaded_cache",
        lambda *args, **kwargs: built,
    )
    monkeypatch.setattr(
        runtime,
        "move_and_bind_m03r_v13_sequence",
        lambda *args, **kwargs: sequence,
    )
    monkeypatch.setattr(runtime, "top2000_m03r_v7_decision_inputs", lambda _: inputs)
    monkeypatch.setattr(
        runtime,
        "replace",
        lambda value, **kwargs: SimpleNamespace(decision_state=kwargs["decision_state"]),
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
            return torch.ones((local_origins.numel(), 1, 4, 8))

    monkeypatch.setattr(runtime, "Top2000M03RV7DecisionStateProvider", _Provider)
    batch = object()
    monkeypatch.setattr(
        runtime,
        "build_m03r_v13_batch_from_origin_states",
        lambda *args, **kwargs: batch,
    )

    def _step(
        policy: object,
        observed_batch: object,
        optimizer: object,
        partition: object,
        update_plan: object,
        paired_input: object,
        **kwargs: object,
    ) -> _Validated:
        assert observed_batch is batch
        captured["update_plan"] = update_plan
        captured["paired_input"] = paired_input
        return _Validated(
            training_update_plan_sha256=update_plan.receipt_sha256,
            paired_input_binding_sha256=paired_input.receipt_sha256,
        )

    monkeypatch.setattr(runtime, "train_m03r_v13_predictive_batch_update", _step)
    policy = SimpleNamespace(
        v13_setting=SimpleNamespace(setting_index=0), source_policy=object()
    )
    result = run_m03r_v13_pretraining_fold_update(
        cache,
        worker,
        schedule,
        geometry,
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
    expected = tuple(
        origin - result.update_plan.episode_start
        for origin in result.update_plan.rank_origins[1]
    )
    assert captured["local_origins"] == expected
    assert min(expected) >= 251
    assert max(expected) <= 373
    assert result.paired_input.cache_sha256 == "b" * 64


def test_v13_runtime_rejects_risk_manifest_before_episode_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_quant.training.top2000_m03r_v13_training_runtime as runtime

    schedule, worker = _schedule_and_worker()
    geometry = render_m03r_v13_fold_geometries(1001)[0]
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
    with pytest.raises(M03RV13TrainingRuntimeError, match="risk"):
        run_m03r_v13_pretraining_fold_update(
            cache,
            worker,
            schedule,
            geometry,
            risk,
            SimpleNamespace(manifest_file_sha256="9" * 64),
            SimpleNamespace(
                v13_setting=SimpleNamespace(setting_index=0), source_policy=object()
            ),
            object(),
            object(),
            completed_updates=1,
            distributed_rank=0,
            distributed_world_size=2,
            device=torch.device("cpu"),
        )
    assert not called
