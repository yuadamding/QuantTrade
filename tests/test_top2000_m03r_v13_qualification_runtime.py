from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from rl_quant.training.top2000_m03r_v13_fold import (
    render_m03r_v13_fold_geometries,
)
from rl_quant.training.top2000_m03r_v13_qualification_runtime import (
    M03RV13QualificationRuntimeError,
    build_m03r_v13_qualification_risk_state,
    run_m03r_v13_fold_qualification,
)


class _Validated(SimpleNamespace):
    def validate(self) -> None:
        return None

    def validate_unmodified(self) -> None:
        return None


def _inputs() -> tuple[Any, ...]:
    geometry = render_m03r_v13_fold_geometries(1001)[5]
    cache = _Validated(cache_sha256="a" * 64, action_hash="b" * 64)
    exposures = _Validated(receipt_sha256="c" * 64)
    risk_source = _Validated(
        cache_sha256="a" * 64,
        action_hash="b" * 64,
        exposures=exposures,
    )
    risk_state = _Validated(
        asset_axis_sha256="b" * 64,
        origin_state_indices=tuple(
            range(
                geometry.qualification_origin_start_inclusive,
                geometry.qualification_origin_stop_exclusive,
            )
        ),
        source_exposure_receipt_sha256="c" * 64,
    )
    policy = SimpleNamespace(
        v13_setting=SimpleNamespace(setting_index=0),
        source_policy=object(),
    )
    loaded = _Validated(
        fold_index=5,
        setting_index=0,
        asset_axis_sha256="b" * 64,
        model_state_sha256="d" * 64,
        checkpoint_file_sha256="e" * 64,
    )
    return geometry, cache, risk_source, risk_state, policy, loaded


def test_v13_qualification_replays_exact_full_context_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_quant.training.top2000_m03r_v13_qualification_runtime as runtime

    geometry, cache, risk_source, risk_state, policy, loaded = _inputs()
    sequence = SimpleNamespace(decision_state=torch.ones((378, 1, 4, 1)))
    built = SimpleNamespace(
        sequence=object(), identity=SimpleNamespace(receipt_sha256="f" * 64)
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(runtime, "model_state_sha256", lambda _: "d" * 64)
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
    monkeypatch.setattr(runtime, "top2000_m03r_v7_decision_inputs", lambda _: object())
    monkeypatch.setattr(
        runtime,
        "replace",
        lambda value, **kwargs: SimpleNamespace(decision_state=kwargs["decision_state"]),
    )

    class _Provider:
        def __init__(self, value: object) -> None:
            assert value is not None

        def replay_origin_states(
            self,
            source_policy: object,
            bound_sequence: object,
            local_origins: torch.Tensor,
        ) -> torch.Tensor:
            captured["local_origins"] = tuple(int(row) for row in local_origins)
            return torch.ones((63, 1, 4, 8))

    monkeypatch.setattr(runtime, "Top2000M03RV7DecisionStateProvider", _Provider)
    batch = _Validated(
        split="qualification",
        fold_index=5,
        objective=SimpleNamespace(setting=SimpleNamespace(setting_index=0)),
        receipt_sha256="1" * 64,
    )
    monkeypatch.setattr(
        runtime,
        "build_m03r_v13_batch_from_origin_states",
        lambda *args, **kwargs: batch,
    )
    trace = _Validated(
        fold_index=5,
        setting_index=0,
        checkpoint_file_sha256="e" * 64,
        checkpoint_model_state_sha256="d" * 64,
        qualification_batch_receipt_sha256="1" * 64,
    )
    monkeypatch.setattr(
        runtime,
        "run_m03r_v13_simple_sleeve",
        lambda *args, **kwargs: trace,
    )
    result = run_m03r_v13_fold_qualification(
        cache,
        geometry,
        risk_source,
        risk_state,
        policy,
        loaded,
        device=torch.device("cpu"),
    )
    result.validate()
    assert captured["local_origins"] == tuple(range(311, 374))


def test_v13_qualification_rejects_model_drift_before_cache_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_quant.training.top2000_m03r_v13_qualification_runtime as runtime

    geometry, cache, risk_source, risk_state, policy, loaded = _inputs()
    called = False

    def _unexpected(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("cache build must not begin")

    monkeypatch.setattr(runtime, "model_state_sha256", lambda _: "9" * 64)
    monkeypatch.setattr(
        runtime,
        "build_top2000_hold30_development_sequence_from_loaded_cache",
        _unexpected,
    )
    with pytest.raises(M03RV13QualificationRuntimeError, match="model"):
        run_m03r_v13_fold_qualification(
            cache,
            geometry,
            risk_source,
            risk_state,
            policy,
            loaded,
            device=torch.device("cpu"),
        )
    assert called is False


def test_v13_risk_state_uses_exact_fold_origins_and_past_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_quant.training.top2000_m03r_v13_qualification_runtime as runtime

    geometry = render_m03r_v13_fold_geometries(1001)[0]
    daily = torch.ones((1001, 4, 5), dtype=torch.float64)
    daily[:, :, 3] = torch.arange(1, 1002, dtype=torch.float64).unsqueeze(1)
    cache = _Validated(
        daily_ohlcv=daily,
        availability=torch.ones((1001, 4), dtype=torch.bool),
        cache_sha256="a" * 64,
        action_hash="b" * 64,
    )
    risk = _Validated()
    captured: dict[str, Any] = {}

    def _build(*args: object, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(runtime, "build_m03r_v9_device_risk_state", _build)
    result = build_m03r_v13_qualification_risk_state(
        cache,
        geometry,
        risk,
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )
    assert result is not None
    assert captured["origin_state_indices"] == tuple(range(469, 532))
    assert torch.equal(
        captured["daily_log_returns"][0],
        torch.zeros(4, dtype=torch.float64),
    )
    assert not bool(captured["return_available"][0].any())
