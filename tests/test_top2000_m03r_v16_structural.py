from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.protocol.hold30_alpha_m03r_v16_top2000_dev import M03R_V16_SETTINGS
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v9_pretraining_runtime import (
    qualify_m03r_v9_origin_risk_exposures,
)
from rl_quant.training.top2000_m03r_v16_policy import (
    Top2000M03RV16PredictivePolicy,
)
from rl_quant.training.top2000_m03r_v16_pretraining_runtime import (
    build_m03r_v16_batch_from_origin_states,
)
from rl_quant.training.top2000_m03r_v16_structural import (
    build_m03r_v16_structural_slab,
    load_m03r_v16_structural_slab,
    qualify_m03r_v16_structural_slab,
    write_m03r_v16_structural_slab,
)


class _Validated(SimpleNamespace):
    def validate(self) -> None:
        return None

    def validate_unmodified(self) -> None:
        return None


def _surfaces() -> tuple[Any, Any, Hold30Sequence]:
    states = 1001
    assets = 14
    daily = torch.ones((states, assets, 5), dtype=torch.float32)
    time = torch.arange(states, dtype=torch.float32).unsqueeze(1)
    slopes = torch.linspace(-2.0e-5, 2.0e-5, assets).unsqueeze(0)
    daily[..., 3] = torch.exp(time * slopes)
    daily[..., 0] = daily[..., 3]
    daily[..., 1] = daily[..., 3] * 1.001
    daily[..., 2] = daily[..., 3] * 0.999
    daily[..., 4] = 1000.0
    availability = torch.ones((states, assets), dtype=torch.bool)
    cache = _Validated(
        daily_ohlcv=daily,
        availability=availability,
        exchange_dates=tuple(f"date-{index:04d}" for index in range(states)),
        action_ids=("CASH", *(f"A{index}" for index in range(1, assets))),
        cache_sha256="a" * 64,
        action_hash="b" * 64,
    )

    loadings = torch.zeros((states, assets, 6), dtype=torch.float64)
    x = torch.linspace(-1.0, 1.0, assets - 1, dtype=torch.float64)
    loadings[:, 1:, 0] = 1.0
    loadings[:, 1:7, 1] = 1.0
    loadings[:, 7:, 2] = 1.0
    loadings[:, 1:, 3] = x
    loadings[:, 1:, 4] = x.square()
    loadings[:, 1:, 5] = x.pow(3)
    weights = torch.ones((states, assets), dtype=torch.float64)
    weights[:, 0] = 0.0
    decision_time = torch.arange(states, dtype=torch.int64) * 86_400_000
    exposures = qualify_m03r_v9_origin_risk_exposures(
        state_start_index=0,
        cash_index=0,
        projector_exposure_names=(
            "sector-a",
            "sector-b",
            "active-beta",
            "style-return",
            "style-volatility",
        ),
        projector_exposure_families=(
            "sector",
            "sector",
            "active-beta",
            "style-risk",
            "style-risk",
        ),
        asset_axis_sha256="b" * 64,
        source_receipt_sha256="c" * 64,
        exposure_loadings=loadings,
        regression_weights=weights,
        decision_timestamp_ms=decision_time,
        exposure_available_timestamp_ms=decision_time[:, None, None]
        .expand(states, assets, 3)
        .clone(),
    )
    risk = _Validated(
        cache_sha256="a" * 64,
        action_hash="b" * 64,
        exposures=exposures,
        receipt_sha256="d" * 64,
    )

    sequence_states = 345
    close = daily[:sequence_states, :, 3]
    returns = close[1:] / close[:-1] - 1.0
    returns[:, 0] = 0.0
    initial = torch.zeros((1, assets), dtype=torch.float32)
    initial[0, 0] = 0.87
    initial[0, 1:] = 0.01
    sequence_available = availability[:sequence_states].unsqueeze(1)
    sequence = Hold30Sequence(
        decision_state=daily[:sequence_states].unsqueeze(1),
        asset_returns=returns.unsqueeze(1),
        decision_available=sequence_available,
        fill_membership=sequence_available.clone(),
        fill_availability=sequence_available.clone(),
        benchmark_weights=initial.unsqueeze(0).expand(sequence_states, -1, -1).clone(),
        risk_asset_caps=torch.ones((sequence_states, 1, assets)),
        risk_gross_max=torch.ones((sequence_states, 1)),
        benchmark_net_returns=torch.full((sequence_states - 1, 1), 1.0e-5),
        initial_ledger=CohortLedger.from_weights(
            initial, cash_index=0, initial_age=0, track_initial_units=True
        ),
        cost_rate=0.002,
        axis_id="b" * 64,
    )
    return cache, risk, sequence


def _identities() -> dict[str, str]:
    return {
        "cache_manifest_sha256": "e" * 64,
        "source_manifest_sha256": "f" * 64,
        "operator_source_sha256": "0" * 64,
        "risk_artifact_file_sha256": "1" * 64,
        "risk_source_manifest_file_sha256": "2" * 64,
        "projector_manifest_file_sha256": "3" * 64,
        "projector_manifest_sha256": "4" * 64,
        "projector_binding_sha256": "5" * 64,
    }


def test_v16_package_owned_structural_slab_round_trips_and_binds_package(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import rl_quant.training.top2000_m03r_v16_structural as structural

    monkeypatch.setattr(structural, "scheduled_m03r_v16_origins", lambda: (251, 252))
    cache, risk, _sequence = _surfaces()
    slab = build_m03r_v16_structural_slab(cache, risk, **_identities())
    assert slab.receipt.scheduled_origin_count == 2
    assert len(slab.origins) == 2
    assert all(
        value.economic_targets[0].shape == value.economic_targets[2].shape
        for value in slab.origins
    )
    slab.receipt.validate_for_package(
        cache_sha256="a" * 64,
        asset_axis_sha256="b" * 64,
        risk_source_receipt_sha256="d" * 64,
        exposure_receipt_sha256=risk.exposures.receipt_sha256,
        **_identities(),
    )
    with pytest.raises(Exception, match="exact package"):
        replace(slab.receipt, source_manifest_sha256="9" * 64).validate_for_package(
            cache_sha256="a" * 64,
            asset_axis_sha256="b" * 64,
            risk_source_receipt_sha256="d" * 64,
            exposure_receipt_sha256=risk.exposures.receipt_sha256,
            **_identities(),
        )

    path = tmp_path / "v16-structural-slab.pt"
    file_sha256 = write_m03r_v16_structural_slab(path, slab)
    loaded = load_m03r_v16_structural_slab(
        path,
        expected_file_sha256=file_sha256,
        expected_receipt_sha256=slab.receipt.receipt_sha256,
    )
    assert loaded.receipt == slab.receipt
    assert loaded.origin(251).receipt_sha256 == slab.origin(251).receipt_sha256


def test_v16_batch_consumes_slab_without_rebuilding_qr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_quant.training.top2000_m03r_v16_pretraining_runtime as runtime
    import rl_quant.training.top2000_m03r_v16_structural as structural

    monkeypatch.setattr(structural, "scheduled_m03r_v16_origins", lambda: (251,))
    cache, risk, sequence = _surfaces()
    slab = build_m03r_v16_structural_slab(cache, risk, **_identities())
    validated_slab = qualify_m03r_v16_structural_slab(slab)

    policy = Top2000M03RV16PredictivePolicy(
        0,
        token_dim=16,
        raw_stock_chunk=8,
        activation_checkpointing=False,
    )
    origin_states = torch.randn((1, 1, sequence.num_assets, 16))
    live = build_m03r_v16_batch_from_origin_states(
        policy,
        M03R_V16_SETTINGS[0],
        origin_states,
        sequence,
        torch.tensor([251]),
        sequence_global_state_start=0,
        split="training",
        split_start_inclusive=0,
        split_stop_exclusive=345,
        fold_index=0,
        source_array_sha256="6" * 64,
        asset_axis_sha256="b" * 64,
        origin_risk_exposures=risk.exposures,
    )

    def _forbidden(**_kwargs: Any) -> Any:
        raise AssertionError("QR operator rebuilt in the training hot path")

    monkeypatch.setattr(runtime, "build_m03r_v15_residual_operator", _forbidden)
    batch = build_m03r_v16_batch_from_origin_states(
        policy,
        M03R_V16_SETTINGS[0],
        origin_states,
        sequence,
        torch.tensor([251]),
        sequence_global_state_start=0,
        split="training",
        split_start_inclusive=0,
        split_stop_exclusive=345,
        fold_index=0,
        source_array_sha256="6" * 64,
        asset_axis_sha256="b" * 64,
        origin_risk_exposures=risk.exposures,
        structural_slab=validated_slab,
    )
    assert batch.structural_slab_receipt_sha256 == validated_slab.receipt_sha256
    assert torch.equal(
        batch.objective.selection_target_economic[0].cpu(),
        slab.origin(251).economic_targets[0],
    )
    assert torch.allclose(
        batch.objective.selection_target_economic,
        live.objective.selection_target_economic,
        rtol=2.0e-5,
        atol=2.0e-7,
    )


def test_v16_validated_slab_lookup_does_not_repeat_global_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_quant.training.top2000_m03r_v16_structural as structural

    monkeypatch.setattr(structural, "scheduled_m03r_v16_origins", lambda: (251,))
    cache, risk, _sequence = _surfaces()
    slab = build_m03r_v16_structural_slab(cache, risk, **_identities())
    authority = qualify_m03r_v16_structural_slab(slab)

    def _forbidden(_self: Any) -> None:
        raise AssertionError("deep slab validation entered the optimizer hot path")

    monkeypatch.setattr(type(slab), "validate", _forbidden)
    assert authority.origin(251).origin_state_index == 251
    first = authority.device_origin(251, torch.device("cpu"))
    second = authority.device_origin(251, torch.device("cpu"))
    assert first is second


def test_v16_phase_slab_authority_rejects_out_of_view_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_quant.training.top2000_m03r_v16_structural as structural

    monkeypatch.setattr(structural, "scheduled_m03r_v16_origins", lambda: (251, 252))
    cache, risk, _sequence = _surfaces()
    slab = build_m03r_v16_structural_slab(cache, risk, **_identities())
    authority = qualify_m03r_v16_structural_slab(slab)
    training_view = replace(
        authority,
        access_mode="training",
        allowed_origin_indices=frozenset({251}),
    )
    assert training_view.origin(251).origin_state_index == 251
    with pytest.raises(structural.M03RV16StructuralError, match="forbids"):
        training_view.origin(252)
