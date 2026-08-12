from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rl_quant.envs.hold30 import CohortLedger
from rl_quant.training.hold30_runtime import Hold30Sequence
from rl_quant.training.top2000_m03r_v9_policy import M03RV9AlphaDistribution
from rl_quant.training.top2000_m03r_v11_runtime import (
    M03RV11RuntimeError,
    run_m03r_v11_simple_sleeve,
)


class _RiskState:
    def __init__(self, assets: int) -> None:
        self.asset_axis_sha256 = "a" * 64
        self.origin_state_indices = (70, 71, 72)
        self.manifest_sha256 = "b" * 64
        self.state_sha256 = "c" * 64
        self.source_exposure_receipt_sha256 = "f" * 64
        self.cash_index = 0
        self.covariance_factor = torch.zeros((3, assets, 2), dtype=torch.float64)
        self.specific_variance = torch.zeros((3, assets), dtype=torch.float64)

    def validate(self) -> None:
        return None

    def require_fast_identity(self, **kwargs: object) -> None:
        assert kwargs["sequence_asset_axis_sha256"] == self.asset_axis_sha256
        assert kwargs["checkpoint_asset_axis_sha256"] == self.asset_axis_sha256
        assert kwargs["expected_manifest_sha256"] == self.manifest_sha256


class _Operator:
    def __init__(self, origin: int, assets: int) -> None:
        self.origin_state_index = origin
        self.asset_axis_sha256 = "a" * 64
        self.source_exposure_receipt_sha256 = "f" * 64
        self.qualified_asset_mask = torch.ones(assets, dtype=torch.bool)
        self.qualified_asset_mask[0] = False
        self.receipt_sha256 = f"{origin - 69:x}" * 64

    def validate(self) -> None:
        return None


def _sequence() -> Hold30Sequence:
    transitions, assets = 3, 4
    benchmark = torch.full((transitions + 1, 1, assets), 0.3, dtype=torch.float64)
    benchmark[:, :, 0] = 0.1
    returns = torch.zeros((transitions, 1, assets), dtype=torch.float64)
    available = torch.ones((transitions + 1, 1, assets), dtype=torch.bool)
    caps = torch.ones_like(benchmark)
    return Hold30Sequence(
        decision_state=torch.zeros((transitions + 1, 1, assets, 1)),
        asset_returns=returns,
        decision_available=available,
        fill_membership=available,
        fill_availability=available,
        benchmark_weights=benchmark,
        risk_asset_caps=caps,
        risk_gross_max=torch.ones((transitions + 1, 1), dtype=torch.float64),
        benchmark_net_returns=torch.zeros((transitions, 1), dtype=torch.float64),
        initial_ledger=CohortLedger.from_staggered_endowment(
            benchmark[0],
            cash_index=0,
            youngest_age=0,
            oldest_age=29,
            track_initial_units=False,
        ),
        axis_id="a" * 64,
    )


def _distributions() -> tuple[M03RV9AlphaDistribution, ...]:
    rows = []
    for index in range(3):
        means = torch.zeros((1, 4, 4), dtype=torch.float64)
        means[0, 1:, 2] = torch.tensor([0.01, -0.01, 0.02]) + index * 0.001
        log_scale = torch.full_like(means, -4.0)
        rows.append(
            M03RV9AlphaDistribution(
                mean_by_horizon=means,
                log_scale_by_horizon=log_scale,
                selected_horizon_sessions=30,
                selected_mean=means[..., 2],
                selected_scale=torch.exp(log_scale[..., 2]),
            )
        )
    return tuple(rows)


def test_v11_sleeve_routes_exact_shared_operators_and_v3_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_quant.training.top2000_m03r_v11_runtime as runtime

    sequence = _sequence()
    risk = _RiskState(4)
    operators = tuple(_Operator(70 + index, 4) for index in range(3))
    applied: list[str] = []

    def _apply(value: torch.Tensor, operator: _Operator) -> object:
        applied.append(operator.receipt_sha256)
        return SimpleNamespace(residual=value.clone())

    def _project(requested: torch.Tensor, *args: object, **kwargs: object) -> object:
        return SimpleNamespace(
            projected_weights=requested,
            requested_to_executed_retention=torch.ones(
                requested.shape[0], dtype=requested.dtype
            ),
        )

    def _proposal(anchor: torch.Tensor, *args: object, **kwargs: object) -> object:
        assert kwargs["selected_horizon_sessions"] == 30
        return SimpleNamespace(requested_weights=anchor.clone())

    monkeypatch.setattr(runtime, "apply_m03r_v11_residual_operator", _apply)
    monkeypatch.setattr(runtime, "project_m03r_v9_active_book", _project)
    monkeypatch.setattr(runtime, "build_cost_aware_active_proposal_v3", _proposal)
    trace = run_m03r_v11_simple_sleeve(
        sequence,
        _distributions(),
        operators,  # type: ignore[arg-type]
        risk,  # type: ignore[arg-type]
        setting_index=1,
        fold_index=2,
        selected_horizon_sessions=30,
        state_start_index=70,
        checkpoint_file_sha256="d" * 64,
        checkpoint_model_state_sha256="e" * 64,
        checkpoint_asset_axis_sha256="a" * 64,
        source_receipt_sha256="f" * 64,
        benchmark_gross_returns=torch.zeros(3, dtype=torch.float64),
        benchmark_one_way_turnover=torch.zeros(3, dtype=torch.float64),
    )
    trace.validate()
    assert tuple(applied) == trace.signal_operator_receipt_sha256
    assert trace.v11_shared_residual_operator_used
    assert trace.v11_cost_aware_allocator_used
    assert not trace.imported_v9_signal_projector_used


def test_v11_sleeve_rejects_operator_from_wrong_origin_before_trade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rl_quant.training.top2000_m03r_v11_runtime as runtime

    valid_operators = tuple(_Operator(70 + index, 4) for index in range(3))
    operators = (_Operator(99, 4), *valid_operators[1:])
    called = False

    def _unexpected(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("projection must not run")

    monkeypatch.setattr(runtime, "project_m03r_v9_active_book", _unexpected)
    with pytest.raises(M03RV11RuntimeError, match="operator"):
        run_m03r_v11_simple_sleeve(
            _sequence(),
            _distributions(),
            operators,  # type: ignore[arg-type]
            _RiskState(4),  # type: ignore[arg-type]
            setting_index=1,
            fold_index=2,
            selected_horizon_sessions=30,
            state_start_index=70,
            checkpoint_file_sha256="d" * 64,
            checkpoint_model_state_sha256="e" * 64,
            checkpoint_asset_axis_sha256="a" * 64,
            source_receipt_sha256="f" * 64,
            benchmark_gross_returns=torch.zeros(3, dtype=torch.float64),
            benchmark_one_way_turnover=torch.zeros(3, dtype=torch.float64),
        )
    assert not called
