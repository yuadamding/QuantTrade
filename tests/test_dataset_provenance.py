from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rl_quant.datasets.provenance import (
    declared_universe_actions,
    inspect_dataset_provenance,
    point_in_time_membership,
    source_symbol_to_action_index,
)


def _manifest(tmp_path, **updates):
    payload = {
        "first_window": "2022-01-03_to_2022-01-06",
        "universe_selection_date": "2021-12-31",
        "universe_selection_method": "lagged dollar volume",
        "membership_mode": "static",
    }
    payload.update(updates)
    (tmp_path / "manifest.json").write_text(json.dumps(payload))
    (tmp_path / "universe.json").write_text(json.dumps({
        "cash_index": 0,
        "action_count": 2,
        "actions": ["CASH", "AAA"],
    }))
    partition = tmp_path / "partitions" / "2022-01-03_to_2022-01-06"
    partition.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"date_exchange": ["2022-01-03"]}),
        partition / "bars.parquet",
    )


def test_provenance_rejects_future_selected_universe(tmp_path):
    _manifest(tmp_path, universe_selection_date="2026-06-12")
    report = inspect_dataset_provenance(tmp_path)
    assert not report.reportable
    assert any("after sample start" in error for error in report.errors)


def test_provenance_rejects_non_object_manifest_and_malformed_date(tmp_path):
    _manifest(tmp_path)
    (tmp_path / "manifest.json").write_text("[]")
    report = inspect_dataset_provenance(tmp_path)
    assert not report.reportable
    assert report.errors == ("manifest.json must contain an object",)

    _manifest(tmp_path, universe_selection_date="2021-12-31garbage")
    report = inspect_dataset_provenance(tmp_path)
    assert not report.reportable
    assert any("universe_selection_date is missing or invalid" in error for error in report.errors)


def test_provenance_accepts_declared_prior_static_universe(tmp_path):
    _manifest(tmp_path)
    report = inspect_dataset_provenance(tmp_path)
    assert report.reportable
    assert report.membership_mode == "static"
    assert report.warnings


def test_provenance_rejects_manifest_that_hides_earlier_actual_partition(tmp_path):
    _manifest(tmp_path)
    earlier = tmp_path / "partitions" / "2020-01-02_to_2020-01-05"
    earlier.mkdir(parents=True)
    pq.write_table(
        pa.table({"date_exchange": ["2020-01-02"]}),
        earlier / "bars.parquet",
    )

    report = inspect_dataset_provenance(tmp_path)

    assert not report.reportable
    assert any("does not match earliest actual bars date 2020-01-02" in error for error in report.errors)


def test_provenance_rejects_misnamed_partition_with_earlier_bar_content(tmp_path):
    _manifest(tmp_path)
    partition = tmp_path / "partitions" / "2022-01-03_to_2022-01-06"
    pq.write_table(
        pa.table({"date_exchange": ["2020-01-02"]}),
        partition / "bars.parquet",
    )

    report = inspect_dataset_provenance(tmp_path)

    assert not report.reportable
    assert any("does not match earliest actual bars date 2020-01-02" in error for error in report.errors)


def test_point_in_time_membership_respects_effective_and_available_time(tmp_path):
    table = pa.table({
        "symbol": ["AAA", "AAA", "BBB"],
        "effective_date": ["2022-01-01", "2022-01-03", "2022-01-01"],
        "is_member": [True, False, True],
        "available_timestamp_ms": [100, 350, 250],
    })
    pq.write_table(table, tmp_path / "universe_membership.parquet")
    mask = point_in_time_membership(
        tmp_path,
        ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-03"],
        ["CASH", "AAA", "BBB"],
        np.array([150, 200, 300, 400]),
    )
    assert mask[:, 0].all()
    assert mask[:, 1].tolist() == [True, True, True, False]
    assert mask[:, 2].tolist() == [False, False, True, True]


def test_reserved_cash_ticker_uses_explicit_source_alias(tmp_path):
    (tmp_path / "universe.json").write_text(json.dumps({
        "cash_index": 0,
        "action_count": 2,
        "actions": ["CASH", "EQUITY:CASH"],
        "source_symbol_aliases": {"EQUITY:CASH": "CASH"},
    }))
    pq.write_table(
        pa.table({
            "symbol": ["CASH"],
            "effective_date": ["2022-01-01"],
            "is_member": [True],
            "available_timestamp_ms": [100],
        }),
        tmp_path / "universe_membership.parquet",
    )

    assert declared_universe_actions(tmp_path) == ["CASH", "EQUITY:CASH"]
    assert source_symbol_to_action_index(tmp_path) == {"CASH": 1}
    mask = point_in_time_membership(
        tmp_path,
        ["2022-01-02"],
        ["CASH", "EQUITY:CASH"],
        np.array([200]),
    )
    assert mask.tolist() == [[True, True]]


def test_late_old_membership_event_does_not_block_known_newer_event(tmp_path):
    table = pa.table({
        "symbol": ["AAA", "AAA"],
        "effective_date": ["2022-01-01", "2022-01-02"],
        "is_member": [False, True],
        "available_timestamp_ms": [1_000, 100],
    })
    pq.write_table(table, tmp_path / "universe_membership.parquet")

    mask = point_in_time_membership(
        tmp_path,
        ["2022-01-02", "2022-01-02"],
        ["CASH", "AAA"],
        np.array([200, 1_100]),
    )

    assert mask[:, 1].tolist() == [True, True]


def test_membership_rejects_string_booleans_and_fails_provenance(tmp_path):
    _manifest(tmp_path, membership_mode="rolling")
    pq.write_table(
        pa.table({
            "symbol": ["AAA"],
            "effective_date": ["2022-01-01"],
            "is_member": ["false"],
            "available_timestamp_ms": [100],
        }),
        tmp_path / "universe_membership.parquet",
    )

    report = inspect_dataset_provenance(tmp_path)
    assert not report.reportable
    assert any("is_member must be a boolean" in error for error in report.errors)
    with pytest.raises(ValueError, match="is_member must be a boolean"):
        point_in_time_membership(
            tmp_path,
            ["2022-01-02"],
            ["CASH", "AAA"],
            np.array([200]),
        )


def test_provenance_rejects_empty_or_incomplete_rolling_membership(tmp_path):
    _manifest(tmp_path, membership_mode="rolling")
    empty = pa.table({
        "symbol": pa.array([], type=pa.string()),
        "effective_date": pa.array([], type=pa.string()),
        "is_member": pa.array([], type=pa.bool_()),
        "available_timestamp_ms": pa.array([], type=pa.int64()),
    })
    pq.write_table(empty, tmp_path / "universe_membership.parquet")

    report = inspect_dataset_provenance(tmp_path)

    assert not report.reportable
    assert any("contains no membership events" in error for error in report.errors)
    with pytest.raises(ValueError, match="contains no membership events"):
        point_in_time_membership(
            tmp_path,
            ["2022-01-03"],
            ["CASH", "AAA"],
            np.array([200]),
        )

    pq.write_table(
        pa.table({
            "symbol": ["ZZZ"],
            "effective_date": ["2022-01-01"],
            "is_member": [True],
            "available_timestamp_ms": [100],
        }),
        tmp_path / "universe_membership.parquet",
    )
    report = inspect_dataset_provenance(tmp_path)
    assert not report.reportable
    assert any("no positive event for declared actions: AAA" in error for error in report.errors)
    assert any("contains undeclared actions: ZZZ" in error for error in report.errors)
