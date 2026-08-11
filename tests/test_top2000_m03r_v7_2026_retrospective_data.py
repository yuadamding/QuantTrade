"""Focused tests for the separate 2026 TOP2000 retrospective data boundary."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import rl_quant.evaluation.top2000_m03r_v7_2026_retrospective_data as retrospective
from rl_quant.evaluation.top2000_m03r_v7_2026_retrospective_data import (
    TOP2000_M03R_V7_2026_CONTEXT_STATES,
    TOP2000_M03R_V7_2026_CUTOFF,
    TOP2000_M03R_V7_2026_MAX_TOTAL_STATES,
    TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
    Top2000M03RV72026RetrospectiveDataError,
    Top2000M03RV72026RetrospectiveSourceEvidence,
    compose_top2000_m03r_v7_2026_retrospective_data,
    load_top2000_m03r_v7_2026_retrospective_cache,
    materialize_top2000_m03r_v7_2026_retrospective_cache,
)
from rl_quant.training.hold30_top2000_development import (
    Top2000VerifiedDevelopmentCache,
)
from rl_quant.workflows import top2000_ppo


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _axis_digest(values: tuple[str, ...]) -> str:
    payload = (
        json.dumps(
            list(values),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
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
    return tuple(dates)


def _daily(
    dates: tuple[str, ...],
    actions: tuple[str, ...],
    *,
    start_close: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = len(dates)
    bars = torch.zeros((rows, len(actions), 5), dtype=torch.float64)
    close = start_close + torch.arange(rows, dtype=torch.float64) * 0.01
    bars[:, 1:, 0] = close.unsqueeze(-1)
    bars[:, 1:, 1] = close.unsqueeze(-1) + 1.0
    bars[:, 1:, 2] = close.unsqueeze(-1) - 1.0
    bars[:, 1:, 3] = close.unsqueeze(-1)
    bars[:, 1:, 4] = 1_000_000.0
    available = torch.ones((rows, len(actions)), dtype=torch.bool)
    return bars, available


def _pre2026_cache(
    *,
    dates: tuple[str, ...] | None = None,
    actions: tuple[str, ...] = ("CASH", "A1", "A2", "A3"),
) -> Top2000VerifiedDevelopmentCache:
    dates = dates or _weekdays(dt.date(2024, 12, 2), dt.date(2025, 12, 31))
    bars, available = _daily(dates, actions)
    return Top2000VerifiedDevelopmentCache(
        daily_ohlcv=bars,
        availability=available,
        exchange_dates=dates,
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


def _raw(
    cache: Top2000VerifiedDevelopmentCache,
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    dates = (
        cache.exchange_dates[-1],
        "2026-01-02",
        "2026-01-05",
        "2026-06-22",
        "2026-06-23",
    )
    bars, available = _daily(dates, cache.action_ids, start_close=102.0)
    bars[0] = cache.daily_ohlcv[-1]
    available[0] = cache.availability[-1]
    return bars, available, dates


def _source(raw_dates: tuple[str, ...]) -> Top2000M03RV72026RetrospectiveSourceEvidence:
    return Top2000M03RV72026RetrospectiveSourceEvidence(
        base_dataset_identity=_digest("base"),
        search_identity=_digest("search"),
        lockbox_partition_names_hash=_digest("lockbox"),
        test_identity=_digest("test"),
        test_partition_inventory_sha256=_digest("partitions"),
        manifest_sha256=_digest("manifest"),
        universe_sha256=_digest("universe"),
        training_completion_receipt_sha256=_digest("training-complete"),
        evaluation_contract_sha256=_digest("evaluation-contract"),
        raw_first_exchange_date=raw_dates[0],
        raw_last_exchange_date=raw_dates[-1],
    )


def _compose() -> retrospective.Top2000M03RV72026RetrospectiveData:
    cache = _pre2026_cache()
    bars, available, dates = _raw(cache)
    return compose_top2000_m03r_v7_2026_retrospective_data(
        cache,
        retrospective_daily_ohlcv=bars,
        retrospective_availability=available,
        retrospective_exchange_dates=dates,
        retrospective_action_ids=cache.action_ids,
        source_evidence=_source(dates),
        acknowledgement=TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
    )


def test_retrospective_has_252_state_warmup_and_one_exact_2026_score_suffix() -> None:
    built = _compose()
    identity = built.identity

    assert identity.context_state_rows == TOP2000_M03R_V7_2026_CONTEXT_STATES
    assert identity.state_rows == TOP2000_M03R_V7_2026_CONTEXT_STATES + 4
    assert identity.state_rows <= TOP2000_M03R_V7_2026_MAX_TOTAL_STATES
    assert identity.score_transition_start == 251
    assert identity.score_transition_stop_exclusive == 255
    assert identity.score_transition_rows == 4
    assert built.score_transition_slice == slice(251, 255)
    assert built.score_return_dates == (
        "2026-01-02",
        "2026-01-05",
        "2026-06-22",
        "2026-06-23",
    )
    assert built.exchange_dates[-1] == TOP2000_M03R_V7_2026_CUTOFF.isoformat()
    assert built.sequence.asset_returns[built.score_transition_slice].shape == (4, 1, 4)
    # The first scored return crosses the final pre-2026 state into the first
    # 2026 state.  The chronology is warm and is not reset on January 1.
    expected = (
        built.sequence.decision_state[252, 0, 1, 3]
        / built.sequence.decision_state[251, 0, 1, 3]
        - 1.0
    )
    assert built.sequence.asset_returns[251, 0, 1] == pytest.approx(float(expected))
    assert identity.single_continuous_chronology
    assert identity.state_reset_count_within_2026 == 0


def test_receipt_binds_actions_dates_availability_returns_and_complete_c1() -> None:
    built = _compose()
    identity = built.identity

    assert identity.action_hash == _axis_digest(built.action_ids)
    assert identity.benchmark_trace_sha256 == built.benchmark.trace_sha256
    assert identity.benchmark_weights_sha256 == retrospective._tensor_sha256(
        built.benchmark.weights
    )
    assert identity.benchmark_gross_returns_sha256 == retrospective._tensor_sha256(
        built.benchmark.gross_returns
    )
    assert identity.benchmark_total_turnover_sha256 == retrospective._tensor_sha256(
        built.benchmark.total_one_way_turnover
    )
    assert identity.availability_sha256 == retrospective._tensor_sha256(
        built.sequence.decision_available[:, 0]
    )
    assert identity.asset_returns_sha256 == retrospective._tensor_sha256(
        built.sequence.asset_returns
    )
    assert built.sequence.axis_id == identity.axis_id


def test_future_selected_nonreportable_semantics_fail_closed() -> None:
    built = _compose()

    with pytest.raises(Top2000M03RV72026RetrospectiveDataError, match="nonreportable"):
        replace(built.source_evidence, dataset_reportable=True)
    with pytest.raises(Top2000M03RV72026RetrospectiveDataError, match="semantics"):
        replace(built.identity, scientific_reporting_eligible=True)
    with pytest.raises(Top2000M03RV72026RetrospectiveDataError, match="semantics"):
        replace(built.identity, state_reset_count_within_2026=1)


def test_compose_rejects_action_drift_overlap_drift_and_short_context() -> None:
    cache = _pre2026_cache()
    bars, available, dates = _raw(cache)
    kwargs = {
        "retrospective_daily_ohlcv": bars,
        "retrospective_availability": available,
        "retrospective_exchange_dates": dates,
        "source_evidence": _source(dates),
        "acknowledgement": TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
    }
    with pytest.raises(Top2000M03RV72026RetrospectiveDataError, match="actions"):
        compose_top2000_m03r_v7_2026_retrospective_data(
            cache,
            retrospective_action_ids=("CASH", "A1", "A3", "A2"),
            **kwargs,
        )

    changed = bars.clone()
    changed[0, 1, 3] += 1.0
    with pytest.raises(Top2000M03RV72026RetrospectiveDataError, match="overlap"):
        compose_top2000_m03r_v7_2026_retrospective_data(
            cache,
            retrospective_daily_ohlcv=changed,
            retrospective_availability=available,
            retrospective_exchange_dates=dates,
            retrospective_action_ids=cache.action_ids,
            source_evidence=_source(dates),
            acknowledgement=TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
        )

    short_dates = _weekdays(dt.date(2025, 1, 2), dt.date(2025, 12, 15))[:250]
    short = _pre2026_cache(dates=short_dates)
    short_bars, short_available, short_raw_dates = _raw(short)
    with pytest.raises(Top2000M03RV72026RetrospectiveDataError, match="252"):
        compose_top2000_m03r_v7_2026_retrospective_data(
            short,
            retrospective_daily_ohlcv=short_bars,
            retrospective_availability=short_available,
            retrospective_exchange_dates=short_raw_dates,
            retrospective_action_ids=short.action_ids,
            source_evidence=_source(short_raw_dates),
            acknowledgement=TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
        )


def _write_pre2026_cache(
    path: Path,
    cache: Top2000VerifiedDevelopmentCache,
) -> str:
    payload = {
        "schema_version": 1,
        "feature_cache_version": 1,
        "label": "development-only",
        "development_only": True,
        "bars_only": True,
        "bar_seconds": 300,
        "search_identity": cache.search_identity,
        "base_dataset_identity": _digest("base"),
        "lockbox_partition_names_hash": _digest("lockbox"),
        "cache_identity": cache.cache_identity,
        "actions": cache.action_ids,
        "action_hash": cache.action_hash,
        "exchange_dates": cache.exchange_dates,
        "date_hash": _axis_digest(cache.exchange_dates),
        "daily_ohlcv": cache.daily_ohlcv,
        "availability": cache.availability,
    }
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_materialize_once_then_load_recomputes_all_bound_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _pre2026_cache()
    raw_bars, raw_available, raw_dates = _raw(cache)
    source_cache_path = tmp_path / "pre2026.pt"
    source_cache_sha256 = _write_pre2026_cache(source_cache_path, cache)
    root = tmp_path / "TOP2000"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_reportable": False,
                "membership_mode": "static",
                "universe_selection_date": "2026-06-12",
                "built_at_utc": "2026-06-23",
                "reportability_errors": ["future-selected static universe"],
            }
        )
    )
    (root / "universe.json").write_text(
        json.dumps(
            {
                "cash_index": 0,
                "action_count": len(cache.action_ids),
                "actions": list(cache.action_ids),
            }
        )
    )
    partition = top2000_ppo.PartitionRef(
        name="2025-12-30_to_2026-06-24",
        start="2025-12-30",
        end="2026-06-24",
        source_signature=_digest("full-parquet-content"),
    )
    plan = top2000_ppo.EvaluationPlan(
        protocol_version=1,
        label=top2000_ppo.DEVELOPMENT_LABEL,
        development_only=True,
        base_dataset_identity=_digest("base"),
        search_identity=_digest("search"),
        lockbox_partition_names_hash=_digest("lockbox"),
        test_identity=_digest("test"),
        test=(partition,),
        bar_seconds=300,
    )
    market = top2000_ppo.market_data_from_daily_ohlcv(
        raw_bars,
        raw_available,
        raw_dates,
    )
    calls = {"plan": 0, "aggregate": 0}

    def fake_plan(*_args, **_kwargs):
        calls["plan"] += 1
        return plan

    def fake_load(*_args, **kwargs):
        calls["aggregate"] += 1
        assert kwargs["date_end"] == TOP2000_M03R_V7_2026_CUTOFF
        return market, raw_dates

    monkeypatch.setattr(top2000_ppo, "build_evaluation_plan", fake_plan)
    monkeypatch.setattr(top2000_ppo, "load_market_data", fake_load)
    monkeypatch.setattr(
        retrospective, "declared_universe_actions", lambda _root: cache.action_ids
    )
    output = tmp_path / "retrospective.pt"
    receipt = materialize_top2000_m03r_v7_2026_retrospective_cache(
        root,
        source_cache_path,
        output,
        expected_pre2026_cache_sha256=source_cache_sha256,
        expected_base_dataset_identity=_digest("base"),
        expected_search_identity=_digest("search"),
        expected_lockbox_partition_names_hash=_digest("lockbox"),
        training_completion_receipt_sha256=_digest("training-complete"),
        evaluation_contract_sha256=_digest("evaluation-contract"),
        acknowledgement=TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
    )
    assert calls == {"plan": 1, "aggregate": 1}
    assert receipt["state_rows"] == 256
    assert not receipt["reportable"]

    loaded = load_top2000_m03r_v7_2026_retrospective_cache(
        output,
        expected_cache_sha256=receipt["cache_sha256"],
        acknowledgement=TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
    )
    assert loaded.cache_file_sha256 == receipt["cache_sha256"]
    assert loaded.identity.receipt_sha256 == receipt["data_receipt_sha256"]
    assert loaded.score_return_dates[-1] == "2026-06-23"

    tampered = torch.load(output, weights_only=True)
    tampered["availability"] = tampered["availability"].clone()
    tampered["availability"][252, 1] = False
    torch.save(tampered, output)
    tampered_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(
        Top2000M03RV72026RetrospectiveDataError,
        match="do not reproduce",
    ):
        load_top2000_m03r_v7_2026_retrospective_cache(
            output,
            expected_cache_sha256=tampered_sha256,
            acknowledgement=TOP2000_M03R_V7_2026_RETROSPECTIVE_ACK,
        )
