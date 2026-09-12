"""Behavioral causal/missingness checks for the bars-only feature path."""

import copy
import gzip
import math
from datetime import date, timedelta

import pytest

from rl_quant.data_sources.massive import qt200_aggregate_features_v1 as features
from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport


def source(count=75):
    start = date(2020, 1, 2)
    days = [(start + timedelta(days=i)).isoformat() for i in range(count * 2)
            if (start + timedelta(days=i)).weekday() < 5][:count]
    bars = {d: dict(ticker="AAPL", session_date_et=d, adjusted=False,
                   o=str(100 + i), h=str(102 + i), l=str(99 + i), c=str(101 + i),
                   v=str(1000 + i), n=i + 1, vw=str(100 + i)) for i, d in enumerate(days)}
    return days, bars


def build(days, bars, **kwargs):
    return features.feature_rows(ticker="AAPL", sessions=days, bars=bars, **kwargs)


def test_values_are_real_numeric_lagged_features_with_a_fixed_schema():
    days, bars = source()
    rows = build(days, bars)
    assert len(rows) == len(days)
    assert rows[0]["values"] == [None] * len(features.FEATURE_NAMES)
    assert rows[1]["latest_feature_session"] == days[0]
    assert rows[1]["values"][0] == pytest.approx(math.log(101 / 100))
    assert rows[1]["values"][2] == pytest.approx(2 / 3)
    assert rows[64]["values"][9] == pytest.approx(math.log(164 / 101))
    assert all(rows[64]["observed_mask"]) and not rows[64]["training_eligible"]


def test_future_and_current_day_prices_cannot_change_an_earlier_observation():
    days, bars = source()
    reference = build(days, bars)
    changed = copy.deepcopy(bars)
    for d in days[40:]:
        for key in "ohlc":
            changed[d][key] = str(int(changed[d][key]) * 3)
    actual = build(days, changed)
    assert actual[:41] == reference[:41]
    assert actual[41] != reference[41]


def test_missing_day_does_not_compress_time_or_become_zero_return():
    days, bars = source()
    del bars[days[30]]
    rows = build(days, bars)
    assert rows[31]["values"] == [None] * len(features.FEATURE_NAMES)
    assert rows[32]["values"][6] is None
    assert rows[33]["values"][6] is not None
    assert rows[36]["values"][7] is None
    assert rows[37]["values"][7] is not None
    assert rows[51]["values"][8] is None
    assert rows[52]["values"][8] is not None


def test_action_bridge_is_masked_not_falsely_reported_as_alpha():
    days, bars = source()
    split_day = days[30]
    for d in days[30:]:
        for key in "ohlc":
            bars[d][key] = str(int(bars[d][key]) / 2)
    rows = build(days, bars, event_dates=(split_day,))
    assert rows[31]["values"][0] is not None
    assert rows[31]["values"][6] is None
    assert rows[32]["values"][6] is not None
    assert rows[35]["values"][7] is None
    assert rows[36]["values"][7] is not None


def test_known_issue_conflicts_cannot_enter_feature_windows():
    days, bars = source()
    rows = build(days, bars, conflict_dates=(days[30],))
    assert not any(rows[31]["observed_mask"])
    assert rows[32]["values"][6] is None
    assert rows[64]["values"][9] is None


def test_zero_volume_differs_from_missing_and_optional_fields_are_not_invented():
    days, bars = source()
    bars[days[0]].update(v="0", n=None, vw=None, h="101", l="101", o="101", c="101")
    rows = build(days, bars)
    assert rows[1]["values"][3] == 0 and rows[1]["observed_mask"][3]
    assert rows[1]["values"][2] is rows[1]["values"][4] is rows[1]["values"][5] is None


def test_normalization_excludes_validation_and_uses_population_scale():
    days, bars = source()
    rows = build(days, bars)
    fit = days[64:69]
    reference = features.fit_normalizer(rows, fit_sessions=fit, heldout_start=days[69])
    modified = copy.deepcopy(rows)
    for row in modified[69:]:
        row["values"] = [1e30] * len(features.FEATURE_NAMES)
    assert features.fit_normalizer(modified, fit_sessions=fit, heldout_start=days[69]) == reference
    assert reference["counts"] == [5] * len(features.FEATURE_NAMES)
    assert not reference["training_authorized"]
    with pytest.raises(ValueError, match="held-out"):
        features.fit_normalizer(rows, fit_sessions=days[64:70], heldout_start=days[69])


def test_normalization_rejects_incomplete_duplicate_or_noncausal_support():
    days, bars = source()
    rows = build(days, bars)
    for bad in (rows[:64], rows + [rows[64]]):
        with pytest.raises(ValueError):
            features.fit_normalizer(bad, fit_sessions=days[64:69], heldout_start=days[69])
    rows[64]["latest_feature_session"] = days[64]
    with pytest.raises(ValueError, match="future"):
        features.fit_normalizer(rows, fit_sessions=days[64:69], heldout_start=days[69])


@pytest.mark.parametrize("mutation", ["wrong_ticker", "adjusted", "nonfinite", "negative_volume", "off_calendar"])
def test_invalid_feature_sources_fail(mutation):
    days, bars = source()
    row = bars[days[0]]
    if mutation == "wrong_ticker":
        row["ticker"] = "MSFT"
    elif mutation == "adjusted":
        row["adjusted"] = True
    elif mutation == "nonfinite":
        row["c"] = "NaN"
    elif mutation == "negative_volume":
        row["v"] = "-1"
    else:
        bars["2030-01-01"] = row
    with pytest.raises(ValueError):
        build(days, bars)


def persisted_panel(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "panel"
    root.mkdir()
    days, original = source(3)
    bars, index = [], []
    for ticker in features.SYMBOLS:
        bars.extend({**row, "ticker": ticker} for row in original.values())
        index.extend(dict(ticker=ticker, session_date=d, candidate_economic_observation_indices=[],
                          identity_evidence="unresolved") for d in days)
    for name, rows in (("daily-bars.parquet", bars), ("session-index.parquet", index)):
        pq.write_table(pa.Table.from_pylist(rows), root / name)
    bodies = {
        "candidate-economic-observations.json.gz": {"observations": []},
        "decision-clock-plan.json.gz": {"clocks": [dict(decision_session=d,
            latest_feature_session=days[i - 1] if i else None) for i, d in enumerate(days)]},
    }
    for name, obj in bodies.items():
        transport.write_once(root / name, gzip.compress(transport.canonical(obj), mtime=0))
    files = {p.name: dict(path=p.name, bytes=p.stat().st_size, sha256=transport.digest(p.read_bytes()))
             for p in root.iterdir()}
    proof = dict(schema=features.PANEL_SCHEMA, ordered_tickers=list(features.SYMBOLS),
        sessions=len(days), files=files, training_ready=False, source_observation_integration_complete=True)
    body = transport.canonical(proof)
    transport.write_once(root / "COMPLETE.json", body)
    return root, transport.digest(body)


def test_persisted_features_use_parent_files_without_reading_execution_or_writing_sources(tmp_path):
    import pyarrow.parquet as pq

    root, digest = persisted_panel(tmp_path)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    output = tmp_path / "numeric"
    result = features.materialize_features(panel_root=root, panel_sha256=digest, output=output)
    assert result["counts"]["rows"] == 600
    assert result["numeric_feature_preparation_complete"] and not result["future_execution_inputs_read"]
    assert not result["training_ready"] and not result["targets_generated"]
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before
    rows = pq.read_table(output / "features.parquet").to_pylist()
    assert len(rows) == 600 and rows[0]["values"] == [None] * len(features.FEATURE_NAMES)
    assert rows[1]["values"][0] == pytest.approx(math.log(101 / 100))
    for name, proof in result["files"].items():
        assert transport.digest((output / name).read_bytes()) == proof["sha256"]
    with pytest.raises(FileExistsError):
        features.materialize_features(panel_root=root, panel_sha256=digest, output=output)


def test_changed_parent_file_or_completion_is_rejected_before_output_creation(tmp_path):
    root, digest = persisted_panel(tmp_path)
    output = tmp_path / "numeric"
    with pytest.raises(ValueError, match="parent"):
        features.materialize_features(panel_root=root, panel_sha256="f" * 64, output=output)
    assert not output.exists()
    with (root / "daily-bars.parquet").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="file"):
        features.materialize_features(panel_root=root, panel_sha256=digest, output=output)
    assert not output.exists()
