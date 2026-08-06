"""Focused tests for the bounded real-cache Hold-30 development adapter."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest
import torch

from rl_quant.training.hold30_top2000_development import (
    DEVELOPMENT_ACK,
    TOP2000_HOLD30_MAX_STATE_ROWS,
    TOP2000_HOLD30_OBSERVATION_REPRESENTATION,
    TOP2000_HOLD30_SOURCE_BAR_IDENTITY,
    Top2000Hold30DevelopmentError,
    build_top2000_hold30_development_sequence,
    build_top2000_hold30_development_sequence_from_loaded_cache,
    load_verified_top2000_hold30_development_cache,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _cache_axis_digest(values: tuple[str, ...]) -> str:
    encoded = (
        json.dumps(
            list(values),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_cache(
    tmp_path: Path,
    *,
    days: int = 70,
    bar_seconds: int = 300,
    second_day_a1_close: float = 110.0,
) -> tuple[Path, str]:
    dates = tuple(
        (dt.date(2024, 1, 2) + dt.timedelta(days=index)).isoformat()
        for index in range(days)
    )
    actions = ("CASH", "A1", "A2", "A3")
    bars = torch.zeros((days, len(actions), 5), dtype=torch.float64)
    bars[:, 1:, 0] = 100.0
    bars[:, 1:, 1] = 101.0
    bars[:, 1:, 2] = 99.0
    bars[:, 1:, 3] = 100.0
    bars[:, 1:, 4] = 1_000_000.0
    bars[1, 1, 3] = second_day_a1_close
    bars[1, 1, 1] = max(111.0, second_day_a1_close)
    availability = torch.ones((days, len(actions)), dtype=torch.bool)
    # A1 disappears at the third state.  Buy-and-drift must force it to CASH
    # and must not buy it again when it reappears.
    availability[2, 1] = False
    action_hash = _cache_axis_digest(actions)
    date_hash = _cache_axis_digest(dates)
    payload = {
        "schema_version": 1,
        "feature_cache_version": 1,
        "label": "development-only",
        "development_only": True,
        "bars_only": True,
        "bar_seconds": bar_seconds,
        "search_identity": _digest("search"),
        "base_dataset_identity": _digest("base"),
        "lockbox_partition_names_hash": _digest("lockbox"),
        "cache_identity": _digest(f"cache-{days}-{bar_seconds}-{second_day_a1_close}"),
        "actions": actions,
        "action_hash": action_hash,
        "exchange_dates": dates,
        "date_hash": date_hash,
        "daily_ohlcv": bars,
        "availability": availability,
    }
    path = tmp_path / (f"cache-{days}-{bar_seconds}-{int(second_day_a1_close)}.pt")
    torch.save(payload, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _adapt(path: Path, digest: str, *, stop: int = 64):
    return build_top2000_hold30_development_sequence(
        path,
        expected_cache_sha256=digest,
        state_start_index=0,
        state_stop_index_exclusive=stop,
        acknowledgement=DEVELOPMENT_ACK,
    )


def test_real_cache_adapter_is_causal_bounded_and_explicitly_nonpromotable(
    tmp_path: Path,
) -> None:
    path, digest = _write_cache(tmp_path)
    adapted = _adapt(path, digest)

    assert adapted.sequence.decision_state.shape == (64, 1, 4, 5)
    assert adapted.sequence.asset_returns.shape == (63, 1, 4)
    assert adapted.identity.source_bar_identity == TOP2000_HOLD30_SOURCE_BAR_IDENTITY
    assert (
        adapted.identity.observation_representation
        == TOP2000_HOLD30_OBSERVATION_REPRESENTATION
    )
    assert adapted.identity.development_only
    assert adapted.identity.future_selected_universe
    assert not adapted.identity.outer_evaluation_authorized
    assert not adapted.identity.promotion_eligible
    assert adapted.sequence.axis_id == adapted.identity.axis_id
    assert adapted.sequence.initial_ledger.retention_units.count_nonzero() == 0
    ledger = adapted.sequence.initial_ledger.economic_value[0]
    assert torch.allclose(
        ledger[1:, :30].sum(-1),
        adapted.sequence.benchmark_weights[0, 0, 1:],
    )
    assert torch.all(ledger[1:, :30] > 0)
    assert ledger[:, 30:].count_nonzero() == 0

    # Day-zero policy state is copied from day-zero bars only.  Day-one close
    # first appears in the post-decision return tensor.
    assert torch.equal(
        adapted.sequence.decision_state[0, 0],
        torch.load(path, weights_only=True)["daily_ohlcv"][0],
    )
    assert adapted.sequence.asset_returns[0, 0, 1] == pytest.approx(0.10)

    with pytest.raises(Top2000Hold30DevelopmentError, match="bounded"):
        build_top2000_hold30_development_sequence(
            path,
            expected_cache_sha256=digest,
            state_start_index=0,
            state_stop_index_exclusive=TOP2000_HOLD30_MAX_STATE_ROWS + 1,
            acknowledgement=DEVELOPMENT_ACK,
        )


def test_monthly_equal_weight_buy_and_drift_charges_all_turnover_by_cause(
    tmp_path: Path,
) -> None:
    path, digest = _write_cache(tmp_path)
    adapted = _adapt(path, digest)
    trace = adapted.benchmark

    assert torch.allclose(
        trace.weights[0],
        torch.tensor([0.0, 1 / 3, 1 / 3, 1 / 3], dtype=torch.float64),
    )
    # The winner drifts above equal weight; no ordinary rebalance reverses it.
    assert trace.weights[1, 1] > trace.weights[1, 2]
    # A1 is unavailable at state two, so its full drifted weight moves to CASH.
    assert trace.weights[2, 1] == 0.0
    assert trace.weights[2, 0] > 0.0
    assert trace.availability_forced_one_way_turnover[1] > 0.0
    assert trace.costs[1] == pytest.approx(
        0.002 * float(trace.availability_forced_one_way_turnover[1])
    )
    assert trace.net_returns[1] == pytest.approx(
        float(trace.gross_returns[1] - trace.costs[1])
    )
    # Reappearance does not buy A1 until the next monthly rebalance.
    assert trace.weights[3, 1] == 0.0
    february_first = 30
    assert adapted.exchange_dates[february_first] == "2024-02-01"
    assert trace.monthly_rebalance_one_way_turnover[february_first - 1] > 0.0
    assert torch.allclose(
        trace.weights[february_first],
        torch.tensor([0.0, 1 / 3, 1 / 3, 1 / 3], dtype=torch.float64),
    )
    assert torch.allclose(
        trace.total_one_way_turnover,
        trace.availability_forced_one_way_turnover
        + trace.monthly_rebalance_one_way_turnover,
    )


def test_future_row_changes_return_but_not_preceding_decision_state(
    tmp_path: Path,
) -> None:
    first_path, first_hash = _write_cache(tmp_path, second_day_a1_close=110.0)
    first = _adapt(first_path, first_hash)
    second_path, second_hash = _write_cache(tmp_path, second_day_a1_close=120.0)
    second = _adapt(second_path, second_hash)

    assert torch.equal(
        first.sequence.decision_state[0], second.sequence.decision_state[0]
    )
    assert first.sequence.asset_returns[0, 0, 1] == pytest.approx(0.10)
    assert second.sequence.asset_returns[0, 0, 1] == pytest.approx(0.20)


def test_adapter_rejects_cache_without_explicit_five_minute_source(
    tmp_path: Path,
) -> None:
    path, digest = _write_cache(tmp_path, bar_seconds=60)
    with pytest.raises(Top2000Hold30DevelopmentError, match="300-second"):
        _adapt(path, digest)


def test_verified_cache_can_build_multiple_slices_without_rereading_path(
    tmp_path: Path,
) -> None:
    path, digest = _write_cache(tmp_path)
    cache = load_verified_top2000_hold30_development_cache(
        path,
        expected_cache_sha256=digest,
        acknowledgement=DEVELOPMENT_ACK,
    )
    # Deleting the file proves both slices are served by the one verified
    # in-memory cache rather than a hidden SHA/read/torch.load hot path.
    path.unlink()
    first = build_top2000_hold30_development_sequence_from_loaded_cache(
        cache,
        state_start_index=0,
        state_stop_index_exclusive=64,
    )
    second = build_top2000_hold30_development_sequence_from_loaded_cache(
        cache,
        state_start_index=1,
        state_stop_index_exclusive=65,
    )

    assert first.exchange_dates[1:] == second.exchange_dates[:-1]
    assert first.identity.cache_sha256 == second.identity.cache_sha256 == digest


def test_loaded_cache_rejects_in_process_tensor_mutation(tmp_path: Path) -> None:
    path, digest = _write_cache(tmp_path)
    cache = load_verified_top2000_hold30_development_cache(
        path,
        expected_cache_sha256=digest,
        acknowledgement=DEVELOPMENT_ACK,
    )
    cache.daily_ohlcv[0, 1, 0] += 1.0
    with pytest.raises(Top2000Hold30DevelopmentError, match="changed after load"):
        build_top2000_hold30_development_sequence_from_loaded_cache(
            cache,
            state_start_index=0,
            state_stop_index_exclusive=64,
        )
