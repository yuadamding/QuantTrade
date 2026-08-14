from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.protocol.hold30_alpha_m03r_v13_top2000_dev import M03R_V13_SETTINGS
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v13_runtime import (
    _fixed_rank_target,
    _scale_quantiles_tensor,
    run_m03r_v13_simple_sleeve,
)


class _Validated(SimpleNamespace):
    def validate(self) -> None:
        return None


def test_v13_scale_quantiles_keep_probability_tensor_on_cuda_device() -> None:
    """The qualification diagnostic must not construct a CPU q tensor."""

    fake_tensor = pytest.importorskip("torch._subclasses.fake_tensor")
    with fake_tensor.FakeTensorMode():
        scale = torch.arange(8, dtype=torch.float64, device="cuda")
        quantiles = _scale_quantiles_tensor(scale)

    assert quantiles.device == scale.device
    assert quantiles.shape == (5,)


def test_v13_fixed_rank_target_compares_simplex_mass_in_one_dtype() -> None:
    """A valid large float32 book must not fail on reduction roundoff."""

    assets = 1_999
    risky_count = 1_785
    benchmark = torch.zeros(assets, dtype=torch.float32)
    benchmark[1 : risky_count + 1] = 1.0 / risky_count
    caps = torch.zeros_like(benchmark)
    caps[0] = 1.0
    caps[1 : risky_count + 1] = 0.01
    mask = caps > 0.0
    mask[0] = False
    score = torch.arange(assets, dtype=torch.float32)

    # This is the exact failure mode from the first v13 qualification attempt:
    # float32 and float64 reductions of the same anchor differ by much more
    # than the scientific mass-conservation tolerance.
    assert abs(float(benchmark.sum()) - float(benchmark.double().sum())) > 2.0e-10

    requested, active_mass = _fixed_rank_target(benchmark, caps, score, mask)

    assert active_mass == pytest.approx(0.0025)
    assert float(requested.sum()) == pytest.approx(
        float(benchmark.double().sum()), abs=2.0e-10
    )
    assert bool((requested >= -2.0e-12).all())
    assert bool((requested <= caps.double() + 2.0e-12).all())


def test_v13_sleeve_uses_origin_action_mask_and_post_fill_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_quant.training.top2000_m03r_v13_runtime as runtime

    dates = 63
    states = 67
    assets = 4
    axis = "a" * 64
    benchmark = torch.tensor([[0.70, 0.10, 0.10, 0.10]], dtype=torch.float64)
    benchmark_weights = benchmark.unsqueeze(0).expand(states, -1, -1).clone()
    returns = torch.zeros((states - 1, 1, assets), dtype=torch.float64)
    # Transition zero must never be earned.  Every action is instead exposed
    # to the transition at local_origin + 1.
    returns[0, 0] = torch.tensor((0.0, -0.50, 0.50, 0.0))
    returns[1 : dates + 1, 0, 1] = 0.01
    returns[1 : dates + 1, 0, 2] = -0.01
    available = torch.ones((states, 1, assets), dtype=torch.bool)
    sequence = Hold30Sequence(
        decision_state=torch.zeros((states, 1, assets, 1), dtype=torch.float64),
        asset_returns=returns,
        decision_available=available,
        fill_membership=available.clone(),
        fill_availability=available.clone(),
        benchmark_weights=benchmark_weights,
        risk_asset_caps=torch.ones((states, 1, assets), dtype=torch.float64),
        risk_gross_max=torch.ones((states, 1), dtype=torch.float64),
        benchmark_net_returns=torch.zeros((states - 1, 1), dtype=torch.float64),
        initial_ledger=CohortLedger.from_weights(
            benchmark, cash_index=0, initial_age=0, track_initial_units=True
        ),
        axis_id=axis,
    )
    score = torch.tensor((0.0, 3.0, -2.0, -1.0), dtype=torch.float64).repeat(
        dates, 1
    )
    valid = torch.tensor((False, True, True, True)).repeat(dates, 1)
    objective = SimpleNamespace(
        predicted_mean=score,
        predicted_log_scale=torch.full_like(score, -4.0),
        target_log_return=score * 0.001,
        valid=valid,
        setting=M03R_V13_SETTINGS[0],
    )
    action_mask = torch.tensor((False, True, True, True))
    target_mask = torch.tensor((False, False, True, True))
    action_operators = tuple(
        _Validated(
            qualified_asset_mask=action_mask,
            receipt_sha256=f"{index + 1:064x}",
        )
        for index in range(dates)
    )
    target_operators = tuple(
        _Validated(
            qualified_asset_mask=target_mask,
            receipt_sha256=f"{index + 1000:064x}",
        )
        for index in range(dates)
    )
    batch = _Validated(
        objective=objective,
        origin_indices=torch.arange(dates, dtype=torch.int64),
        split="qualification",
        fold_index=0,
        asset_axis_sha256=axis,
        exposure_receipt_sha256="b" * 64,
        source_array_sha256="c" * 64,
        action_residual_operators=action_operators,
        target_residual_operators=target_operators,
        receipt_sha256="d" * 64,
    )
    loaded = _Validated(
        setting_index=0,
        setting_id=M03R_V13_SETTINGS[0].setting_id,
        fold_index=0,
        asset_axis_sha256=axis,
        checkpoint_file_sha256="e" * 64,
        model_state_sha256="f" * 64,
    )
    risk = _Validated(
        origin_state_indices=tuple(range(dates)),
        asset_axis_sha256=axis,
        source_exposure_receipt_sha256="b" * 64,
        manifest_sha256="1" * 64,
        state_sha256="2" * 64,
    )
    risk.require_fast_identity = lambda **_kwargs: None
    monkeypatch.setattr(
        runtime,
        "apply_m03r_v11_residual_operator",
        lambda value, operator: SimpleNamespace(residual=value.clone()),
    )
    monkeypatch.setattr(
        runtime,
        "project_m03r_v9_active_book",
        lambda requested, *_args, **_kwargs: SimpleNamespace(
            projected_weights=requested,
            requested_to_executed_retention=torch.ones(1, dtype=requested.dtype),
        ),
    )

    trace = run_m03r_v13_simple_sleeve(
        sequence,
        batch,  # type: ignore[arg-type]
        risk,  # type: ignore[arg-type]
        loaded,  # type: ignore[arg-type]
        sequence_global_state_start=0,
    )
    assert trace.policy_gross_returns[0] > trace.benchmark_gross_returns[0]
    assert trace.requested_weight_trace[0, 1] > benchmark[0, 1]
    assert trace.chronology_action_count_equals_return_count is True
    assert trace.action_mask_uses_future_availability is False
    assert trace.policy_gross_returns.numel() == dates
    assert trace.net_active_return_by_cost[0].shape == (dates,)
