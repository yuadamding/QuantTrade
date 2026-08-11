"""Focused tests for the receipt-bound pre-2026 execution-factor fit."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import replace
from itertools import pairwise

import pytest
import torch

from rl_quant.evaluation.top2000_m03r_v7_2026_factor_calibration import (
    TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_RULE_SHA256,
    Top2000M03RV72026FactorCalibrationError,
    fit_top2000_m03r_v7_2026_pre_score_factor_calibration,
)
from rl_quant.evaluation.top2000_m03r_v7_2026_retrospective_data import (
    TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
    Top2000M03RV72026RetrospectiveData,
    Top2000M03RV72026RetrospectiveSourceEvidence,
    compose_top2000_m03r_v7_2026_retrospective_data,
)
from rl_quant.training.hold30_top2000_development import (
    Top2000VerifiedDevelopmentCache,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _axis_digest(values: tuple[str, ...]) -> str:
    payload = (
        json.dumps(
            list(values),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _weekdays(start: dt.date, stop: dt.date) -> tuple[str, ...]:
    dates: list[str] = []
    current = start
    while current <= stop:
        if current.weekday() < 5:
            dates.append(current.isoformat())
        current += dt.timedelta(days=1)
    assert all(left < right for left, right in pairwise(dates))
    return tuple(dates)


def _daily(
    dates: tuple[str, ...],
    actions: tuple[str, ...],
    *,
    offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = len(dates)
    assets = len(actions)
    bars = torch.zeros((rows, assets, 5), dtype=torch.float64)
    time = torch.arange(rows, dtype=torch.float64).unsqueeze(-1)
    stock = torch.arange(1, assets, dtype=torch.float64).unsqueeze(0)
    close = 100.0 + offset + time * (0.0025 + stock * 0.0007)
    close = close + torch.sin(time / (3.0 + stock)) * stock * 0.01
    bars[:, 1:, 0] = close
    bars[:, 1:, 1] = close + 1.0
    bars[:, 1:, 2] = close - 1.0
    bars[:, 1:, 3] = close
    bars[:, 1:, 4] = 1_000_000.0 + time * 100.0 + stock * 1_000.0
    available = torch.ones((rows, assets), dtype=torch.bool)
    return bars, available


def _retrospective() -> Top2000M03RV72026RetrospectiveData:
    actions = ("CASH", *(f"A{index}" for index in range(1, 8)))
    pre_dates = _weekdays(dt.date(2024, 11, 1), dt.date(2025, 12, 31))
    pre_bars, pre_available = _daily(pre_dates, actions, offset=0.0)
    cache = Top2000VerifiedDevelopmentCache(
        daily_ohlcv=pre_bars,
        availability=pre_available,
        exchange_dates=pre_dates,
        action_ids=actions,
        cache_sha256=_digest("pre-cache-file"),
        cache_identity=_digest("pre-cache-identity"),
        search_identity=_digest("search"),
        action_hash=_axis_digest(actions),
        bar_seconds=300,
        acknowledgement="I acknowledge TOP2000 results are development-only",
        development_only=True,
        bars_only=True,
    )
    raw_dates = (
        pre_dates[-1],
        "2026-01-02",
        "2026-01-05",
        "2026-06-22",
        "2026-06-23",
    )
    raw_bars, raw_available = _daily(raw_dates, actions, offset=3.0)
    raw_bars[0] = pre_bars[-1]
    raw_available[0] = pre_available[-1]
    source = Top2000M03RV72026RetrospectiveSourceEvidence(
        base_dataset_identity=_digest("base"),
        search_identity=cache.search_identity,
        lockbox_partition_names_hash=_digest("lockbox"),
        test_identity=_digest("test"),
        test_partition_inventory_sha256=_digest("partitions"),
        manifest_sha256=_digest("manifest"),
        universe_sha256=_digest("universe"),
        training_completion_receipt_sha256=_digest("complete"),
        evaluation_contract_sha256=_digest("contract"),
        raw_first_exchange_date=raw_dates[0],
        raw_last_exchange_date=raw_dates[-1],
    )
    built = compose_top2000_m03r_v7_2026_retrospective_data(
        cache,
        retrospective_daily_ohlcv=raw_bars,
        retrospective_availability=raw_available,
        retrospective_exchange_dates=raw_dates,
        retrospective_action_ids=actions,
        source_evidence=source,
        acknowledgement=TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
    )
    return replace(built, cache_file_sha256=_digest("retrospective-cache-file"))


def _expected_loadings(data: Top2000M03RV72026RetrospectiveData) -> torch.Tensor:
    returns = data.sequence.asset_returns[:63, 0]
    available = data.sequence.decision_available[:64, 0].all(0).clone()
    available[0] = False
    benchmark = data.sequence.benchmark_net_returns[:63, 0]
    centered_market = benchmark - benchmark.mean()
    market_variance = centered_market.square().sum().clamp_min(1.0e-12)
    centered_assets = returns - returns.mean(0, keepdim=True)
    beta = (centered_assets * centered_market.unsqueeze(-1)).sum(0) / market_variance
    momentum = torch.log1p(returns.clamp_min(-0.999999)).sum(0)
    volatility = returns.std(0, unbiased=False)
    volume = torch.log1p(
        data.sequence.decision_state[:64, 0, :, 4].clamp_min(0.0)
    ).mean(0)
    values = torch.stack((beta, momentum, volatility, volume), dim=-1)
    values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    selected = values[available]
    expected = (values - selected.mean(0)) / selected.std(
        0, unbiased=False
    ).clamp_min(1.0e-6)
    expected[~available] = 0.0
    expected[0] = 0.0
    return expected


def test_factor_fit_uses_exact_pre2026_prefix_and_training_formula() -> None:
    data = _retrospective()
    fitted = fit_top2000_m03r_v7_2026_pre_score_factor_calibration(data)

    assert fitted.state_start_index == 0
    assert fitted.state_stop_index_exclusive == 64
    assert fitted.transition_start_index == 0
    assert fitted.transition_stop_index_exclusive == 63
    assert fitted.calibration_state_dates == data.exchange_dates[:64]
    assert fitted.calibration_state_dates[-1] < "2026-01-01"
    assert fitted.rule_sha256 == TOP2000_M03R_V7_2026_FACTOR_CALIBRATION_RULE_SHA256
    assert torch.allclose(fitted.loadings, _expected_loadings(data))
    assert torch.equal(fitted.loadings[0], torch.zeros(4, dtype=torch.float64))
    assert fitted.retrospective_data_receipt_sha256 == data.identity.receipt_sha256
    assert fitted.retrospective_cache_file_sha256 == data.cache_file_sha256
    assert not fitted.includes_2026_observation
    assert not fitted.policy_training_authorized


def test_post_prefix_and_2026_observations_cannot_move_fit_or_input_hashes() -> None:
    data = _retrospective()
    original = fit_top2000_m03r_v7_2026_pre_score_factor_calibration(data)
    changed_state = data.sequence.decision_state.clone()
    changed_state[64:, 0, 1:, 4] += 99_000_000.0
    changed_returns = data.sequence.asset_returns.clone()
    changed_returns[63:, 0, 1:] += 0.01
    changed_benchmark = data.sequence.benchmark_net_returns.clone()
    changed_benchmark[63:, 0] += 0.005
    changed_sequence = replace(
        data.sequence,
        decision_state=changed_state,
        asset_returns=changed_returns,
        benchmark_net_returns=changed_benchmark,
    )
    changed = replace(data, sequence=changed_sequence)

    refitted = fit_top2000_m03r_v7_2026_pre_score_factor_calibration(changed)

    assert torch.equal(refitted.loadings, original.loadings)
    assert refitted.input_array_inventory_sha256 == original.input_array_inventory_sha256
    assert refitted.loadings_sha256 == original.loadings_sha256
    assert refitted.receipt_sha256 == original.receipt_sha256


def test_factor_receipt_detects_loading_date_and_array_inventory_drift() -> None:
    fitted = fit_top2000_m03r_v7_2026_pre_score_factor_calibration(_retrospective())
    changed_loadings = fitted.loadings.clone()
    changed_loadings[1, 0] += 1.0
    with pytest.raises(Top2000M03RV72026FactorCalibrationError, match="loading"):
        replace(fitted, loadings=changed_loadings)
    with pytest.raises(Top2000M03RV72026FactorCalibrationError, match="date-axis"):
        replace(
            fitted,
            calibration_state_dates=(
                *fitted.calibration_state_dates[:-1],
                "2026-01-02",
            ),
        )
    with pytest.raises(Top2000M03RV72026FactorCalibrationError, match="inventory"):
        replace(fitted, input_array_inventory_sha256=_digest("changed-inventory"))


def test_factor_fit_requires_materialized_cache_receipt() -> None:
    data = replace(_retrospective(), cache_file_sha256=None)
    with pytest.raises(Top2000M03RV72026FactorCalibrationError, match="cache file"):
        fit_top2000_m03r_v7_2026_pre_score_factor_calibration(data)
