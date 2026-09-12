"""Real persisted bars-to-label preparation; no synthetic qualification flags."""

import gzip
import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from rl_quant.data_sources.massive import qt200_aggregate_features_v1 as features
from rl_quant.data_sources.massive import qt200_aggregate_targets_v1 as targets
from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport


def calendar(count=145):
    start = date(2020, 1, 2)
    return [(start + timedelta(days=i)).isoformat() for i in range(count * 2)
            if (start + timedelta(days=i)).weekday() < 5][:count]


def compute(days, **kwargs):
    return targets.target_rows(ticker="AAPL", sessions=days,
        daily_sessions=kwargs.pop("daily_sessions", set(days)),
        window_open=kwargs.pop("window_open", {d: 100.0 + i for i, d in enumerate(days)}), **kwargs)


def test_actual_next_session_entry_and_nonoverlapping_horizons():
    days = calendar()
    row = compute(days)[0]
    assert row["entry_session"] == days[1]
    assert row["bucket_start_sessions"] == [days[1 + x] for x in targets.BOUNDARIES[:-1]]
    assert row["bucket_end_sessions"] == [days[1 + x] for x in targets.BOUNDARIES[1:]]
    assert row["label_maturity_sessions"] == [days[2 + x] for x in targets.BOUNDARIES[1:]]
    assert row["values"] == pytest.approx([(101 + b) / (101 + a) - 1
        for a, b in zip(targets.BOUNDARIES, targets.BOUNDARIES[1:])])
    assert math.prod(1 + v for v in row["values"]) == pytest.approx(227 / 101)
    assert all(row["observed_mask"]) and not row["training_eligible"]


def test_missing_intermediate_day_does_not_compress_target_horizon():
    days = calendar()
    row = compute(days, daily_sessions=set(days) - {days[4]})[0]
    assert row["values"][0] is not None and row["values"][1] is None
    assert row["reason_bits"][1] == targets.REASONS["daily_path_missing"]
    assert row["values"][2] is not None


@pytest.mark.parametrize("boundary", [1, 3, 6])
def test_candidate_actions_mask_start_interior_and_end(boundary):
    days = calendar()
    row = compute(days, event_dates=[days[boundary]])[0]
    bucket = 0 if boundary == 1 else 1
    assert row["values"][bucket] is None
    assert row["reason_bits"][bucket] & targets.REASONS["candidate_action_unresolved"]


def test_unadjusted_split_jump_is_not_labeled_as_predictable_profit():
    days = calendar()
    prices = {d: (100.0 if i < 4 else 50.0) for i, d in enumerate(days)}
    row = compute(days, window_open=prices, event_dates=[days[4]])[0]
    assert row["values"][1] is None
    assert row["values"][2] == 0.0  # observed zero return is not missing


def test_known_identity_conflicts_and_window_missingness_have_distinct_masks():
    days = calendar()
    prices = {d: 100.0 for d in days if d != days[2]}
    row = compute(days, window_open=prices, conflict_dates=[days[4]])[0]
    assert row["reason_bits"][0] == targets.REASONS["end_window_missing"]
    assert row["reason_bits"][1] == (targets.REASONS["start_window_missing"] | targets.REASONS["known_issue_conflict"])


def test_unmatured_tail_is_retained_in_full_calendar():
    days = calendar(10)
    rows = compute(days)
    assert len(rows) == 10
    assert rows[-1]["entry_session"] is None
    assert all(v is None for v in rows[-1]["values"])
    assert rows[-3]["reason_bits"][0] == targets.REASONS["maturity_outside_calendar"]
    assert rows[-3]["label_maturity_sessions"][0] is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0, -1, True])
def test_invalid_target_prices_fail_closed(bad):
    days = calendar()
    with pytest.raises(ValueError, match="price"):
        compute(days, window_open={days[1]: bad})


def test_off_calendar_duplicate_and_out_of_order_dates_rejected():
    days = calendar()
    for invalid in (days + [days[-1]], list(reversed(days))):
        with pytest.raises(ValueError, match="calendar"):
            compute(invalid)
    with pytest.raises(ValueError, match="calendar"):
        compute(days, daily_sessions=set(days) | {"2010-01-01"})


@pytest.fixture(scope="module")
def persisted_inputs(tmp_path_factory):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path_factory.mktemp("bars-target-fixture")
    panel = root / "panel"
    panel.mkdir()
    days = calendar(80)
    bars, minutes, index = [], [], []
    for ticker in features.SYMBOLS:
        for i, day in enumerate(days):
            raw = dict(ticker=ticker, session_date_et=day, adjusted=False,
                       o=str(100 + i), h=str(102 + i), l=str(99 + i), c=str(101 + i),
                       v="10000", n=100, vw=str(100 + i))
            start = len(minutes)
            for slot in range(35, 45):
                timestamp = int(datetime.fromisoformat(day + f"T09:{slot}:00").replace(
                    tzinfo=ZoneInfo("America/New_York")).timestamp()) * 1000
                minutes.append({**raw, "bar_start_ms": timestamp})
            index.append(dict(ticker=ticker, session_date=day, daily_row=len(bars),
                window_row_start=start, window_row_count=10,
                candidate_economic_observation_indices=[], identity_evidence="unresolved"))
            bars.append(raw)
    for name, rows in (("daily-bars.parquet", bars), ("minute-window-bars.parquet", minutes),
                       ("session-index.parquet", index)):
        pq.write_table(pa.Table.from_pylist(rows), panel / name, compression="zstd")
    bodies = {
        "candidate-economic-observations.json.gz": {"observations": []},
        "decision-clock-plan.json.gz": {"clocks": [dict(decision_session=day,
            latest_feature_session=days[i - 1] if i else None,
            execution_session=days[i + 1] if i + 1 < len(days) else None) for i, day in enumerate(days)]},
    }
    for name, obj in bodies.items():
        transport.write_once(panel / name, gzip.compress(transport.canonical(obj), mtime=0))
    files = {p.name: dict(path=p.name, bytes=p.stat().st_size, sha256=transport.digest(p.read_bytes()))
             for p in panel.iterdir()}
    proof = dict(schema=targets.PANEL_SCHEMA, ordered_tickers=list(features.SYMBOLS),
        sessions=len(days), files=files, training_ready=False, source_observation_integration_complete=True,
        counts=dict(minute_rows=len(minutes), daily_rows=len(bars)))
    transport.write_once(panel / "COMPLETE.json", transport.canonical(proof))
    pdigest = transport.digest((panel / "COMPLETE.json").read_bytes())
    numeric = root / "features"
    features.materialize_features(panel_root=panel, panel_sha256=pdigest, output=numeric)
    fdigest = transport.digest((numeric / "COMPLETE.json").read_bytes())
    output = root / "targets"
    result = targets.materialize_targets(panel_root=panel, panel_sha256=pdigest,
        feature_root=numeric, feature_sha256=fdigest, output=output)
    tdigest = transport.digest((output / "COMPLETE.json").read_bytes())
    return dict(panel_root=panel, panel_sha256=pdigest, feature_root=numeric,
        feature_sha256=fdigest, target_root=output, target_sha256=tdigest, result=result, days=days)


def test_actual_persisted_targets_and_feature_links(persisted_inputs):
    import pyarrow.parquet as pq

    p = persisted_inputs
    result = p["result"]
    assert result["counts"]["rows"] == 16000
    assert result["targets_generated"] and result["feature_inputs_unchanged"]
    assert not result["training_ready"] and not result["native_v5_qualified"]
    rows = pq.read_table(p["target_root"] / "targets.parquet").to_pylist()
    assert [r["feature_row_index"] for r in rows] == list(range(16000))
    assert rows[0]["values"][0] == pytest.approx(102 / 101 - 1)
    assert all(row["training_eligible"] is False for row in rows)
    spec = targets.target_specification()
    assert not spec["price_proxy_is_vwap"] and not spec["price_proxy_is_a_fill"]
    assert not spec["transaction_costs_in_label"]


def test_replay_preserves_inputs_and_produces_identical_numeric_content(persisted_inputs, tmp_path):
    p = persisted_inputs
    before = {str(path): path.read_bytes() for root in (p["panel_root"], p["feature_root"])
              for path in root.iterdir()}
    result = targets.materialize_targets(**{k: p[k] for k in
        ("panel_root", "panel_sha256", "feature_root", "feature_sha256")}, output=tmp_path / "replay")
    assert result == p["result"]
    assert all(__import__("pathlib").Path(path).read_bytes() == body for path, body in before.items())
    with pytest.raises(FileExistsError):
        targets.materialize_targets(**{k: p[k] for k in
            ("panel_root", "panel_sha256", "feature_root", "feature_sha256")}, output=tmp_path / "replay")


def test_parent_hash_mismatch_blocks_before_publication(persisted_inputs, tmp_path):
    p = persisted_inputs
    with pytest.raises(ValueError, match="parent"):
        targets.materialize_targets(panel_root=p["panel_root"], panel_sha256=p["panel_sha256"],
            feature_root=p["feature_root"], feature_sha256="f" * 64, output=tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()
