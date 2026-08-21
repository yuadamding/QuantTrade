from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import rl_quant.data_sources.polygon_pit_alpha as polygon
from rl_quant.data_sources.polygon_pit_alpha import (
    EXPECTED_NORMAL_SESSION_INTERVALS,
    OrganizedPolygonShard,
    PolygonPITAlphaConversionError,
    aggregate_polygon_second_bars_to_five_minutes,
    audit_organized_polygon_for_pit_alpha,
    build_exchange_session_authority,
    convert_symbol_day_to_five_minute_staging,
    iter_corporate_action_candidates,
    load_exchange_session_authority,
    load_five_minute_staging_publication,
    resolve_symbol_day_source,
    write_exchange_session_authority,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)

EASTERN = ZoneInfo("America/New_York")
MANIFEST_FIELDS = (
    "symbol",
    "date",
    "status",
    "rows",
    "output_path",
    "output_size",
    "sha256",
    "elapsed_seconds",
    "error",
)


def _timestamp_ms(day: str, hour: int, minute: int, second: int = 0) -> int:
    value = datetime.fromisoformat(
        f"{day}T{hour:02d}:{minute:02d}:{second:02d}"
    ).replace(tzinfo=EASTERN)
    return int(value.timestamp() * 1_000)


def _second_frame(*, day: str = "2024-01-02", symbol: str = "AAA") -> pd.DataFrame:
    rows = (
        (9, 30, 1, 10.0, 10.5, 9.9, 10.2, 100.0, 10.1, 2),
        (9, 34, 59, 10.2, 11.0, 10.1, 10.9, 50.0, 10.8, 3),
        (9, 45, 0, 12.0, 12.5, 11.5, 12.2, 40.0, 12.1, 1),
    )
    return pd.DataFrame(
        {
            "symbol": [symbol] * len(rows),
            "timestamp_ms": [_timestamp_ms(day, *row[:3]) for row in rows],
            "open": [row[3] for row in rows],
            "high": [row[4] for row in rows],
            "low": [row[5] for row in rows],
            "close": [row[6] for row in rows],
            "volume": [row[7] for row in rows],
            "vwap": [row[8] for row in rows],
            "transactions": [row[9] for row in rows],
            "adjusted": [True] * len(rows),
            "timespan": ["second"] * len(rows),
            "multiplier": [1] * len(rows),
        }
    )


def _session(
    *,
    day: str = "2024-01-02",
    close_hour: int = 16,
    special_reason: str | None = None,
) -> polygon.PolygonExchangeSessionAuthority:
    return build_exchange_session_authority(
        session_date=day,
        exchange="XNYS",
        open_at_ms=_timestamp_ms(day, 9, 30),
        close_at_ms=_timestamp_ms(day, close_hour, 0),
        special_session_reason=special_reason,
        assumed_availability_lag_ms=2_000,
        calendar_source_receipt_sha256="c" * 64,
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _organized_shard(
    tmp_path: Path,
    *,
    manifest_sha: bool = True,
    stale_path: bool = False,
    extra_covariate: bool = False,
    identity_transition: bool = False,
    source_status: str = "downloaded",
) -> OrganizedPolygonShard:
    bars = tmp_path / "bars"
    covariates = tmp_path / "covariates"
    bars.mkdir(parents=True)
    covariates.mkdir()
    universe = tmp_path / "universe.txt"
    universe.write_text("AAA\nBBB\n")
    (bars / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "start": "2024-01-01",
                "end_exclusive": "2024-01-03",
                "symbols": 2,
                "market_weekdays": 1,
                "remaining_symbol_days": 0,
            }
        )
    )
    canonical = bars / "AAA" / "2024" / "01" / "2024-01-02.parquet"
    canonical.parent.mkdir(parents=True)
    _second_frame().to_parquet(canonical, index=False)
    source_sha = hashlib.sha256(canonical.read_bytes()).hexdigest()
    output_path = (
        "old-root/AAA/2024/01/2024-01-02.parquet" if stale_path else str(canonical)
    )
    _write_manifest(
        bars / "manifest.csv",
        [
            {
                "symbol": "AAA",
                "date": "2024-01-02",
                "status": source_status,
                "rows": len(_second_frame()),
                "output_path": output_path,
                "output_size": canonical.stat().st_size,
                "sha256": source_sha if manifest_sha else "",
            },
            {
                "symbol": "BBB",
                "date": "2024-01-02",
                "status": "empty",
                "rows": 0,
                "output_path": str(bars / "BBB" / "2024" / "01" / "2024-01-02.parquet"),
                "output_size": 0,
                "sha256": "",
            },
        ],
    )
    aaa_rows: list[dict[str, object]] = [
        {
            "asof_date": "2024-01-01",
            "ticker": "AAA",
            "share_class_figi": "FIGI-1",
            "composite_figi": "COMP-1",
            "cik": "1",
        }
    ]
    if identity_transition:
        aaa_rows.append(
            {
                "asof_date": "2024-02-01",
                "ticker": "AAA",
                "share_class_figi": "FIGI-2",
                "composite_figi": "COMP-2",
                "cik": "2",
            }
        )
    _write_jsonl(covariates / "AAA" / "overview_snapshots.jsonl", aaa_rows)
    _write_jsonl(
        covariates / "BBB" / "overview_snapshots.jsonl",
        [
            {
                "asof_date": "2024-01-01",
                "ticker": "BBB",
                "share_class_figi": "FIGI-B",
                "composite_figi": "COMP-B",
                "cik": "2",
            }
        ],
    )
    if extra_covariate:
        (covariates / "CCC").mkdir()
    _write_jsonl(
        covariates / "AAA" / "dividends.jsonl",
        [
            {
                "id": "div-1",
                "declaration_date": "2024-02-01",
                "ex_dividend_date": "2024-03-01",
                "record_date": "2024-03-04",
                "pay_date": "2024-03-15",
                "cash_amount": 0.25,
            }
        ],
    )
    return OrganizedPolygonShard(
        name="fixture",
        second_aggs_root=bars,
        covariates_root=covariates,
        universe_tickers_file=universe,
        universe_asof="2024-01-02",
    )


def test_five_minute_aggregation_preserves_sparse_missingness() -> None:
    result = aggregate_polygon_second_bars_to_five_minutes(_second_frame(), _session())

    assert list(result["interval_index"]) == [0, 3]
    assert len(result) < EXPECTED_NORMAL_SESSION_INTERVALS
    first = result.iloc[0]
    assert first["open"] == 10.0
    assert first["high"] == 11.0
    assert first["low"] == 9.9
    assert first["close"] == 10.9
    assert first["volume"] == 150.0
    assert first["transactions"] == 5
    assert first["source_row_count"] == 2
    assert first["vwap"] == pytest.approx((10.1 * 100.0 + 10.8 * 50.0) / 150.0)
    assert first["economic_interval_end_ms"] == _timestamp_ms("2024-01-02", 9, 35)
    assert first["assumed_strategy_available_at_ms"] == _timestamp_ms(
        "2024-01-02", 9, 35, 2
    )


def test_early_close_has_42_scheduled_bins_and_filters_outside_rows() -> None:
    frame = _second_frame()
    premarket = frame.iloc[[0]].assign(timestamp_ms=_timestamp_ms("2024-01-02", 8, 0))
    after_close = frame.iloc[[0]].assign(
        timestamp_ms=_timestamp_ms("2024-01-02", 13, 0)
    )
    frame = pd.concat([premarket, frame, after_close], ignore_index=True)
    session = _session(close_hour=13, special_reason="scheduled early close")

    result = aggregate_polygon_second_bars_to_five_minutes(frame, session)

    assert session.scheduled_interval_count == 42
    assert list(result["interval_index"]) == [0, 3]
    assert int(result["interval_index"].max()) < 42


def test_frame_spanning_two_exchange_dates_is_rejected() -> None:
    next_day = _second_frame(day="2024-01-03").iloc[[0]]
    frame = pd.concat([_second_frame(), next_day], ignore_index=True)

    with pytest.raises(PolygonPITAlphaConversionError, match="exchange date"):
        aggregate_polygon_second_bars_to_five_minutes(frame, _session())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "duplicate timestamps"),
        (lambda frame: frame.assign(symbol=["AAA", "BBB", "AAA"]), "one symbol-day"),
        (lambda frame: frame.assign(high=frame["low"] - 1), "OHLC geometry"),
        (lambda frame: frame.assign(volume=-1), "volume cannot be negative"),
        (lambda frame: frame.assign(close=float("nan")), "must be finite"),
        (lambda frame: frame.assign(adjusted=False), "requires adjusted bars"),
        (lambda frame: frame.assign(timespan="minute"), "requires second bars"),
        (lambda frame: frame.assign(multiplier=5), "requires multiplier=1"),
    ],
)
def test_invalid_second_bar_inputs_fail_closed(
    mutate: Callable[[pd.DataFrame], pd.DataFrame], message: str
) -> None:
    frame = mutate(_second_frame())
    with pytest.raises(PolygonPITAlphaConversionError, match=message):
        aggregate_polygon_second_bars_to_five_minutes(frame, _session())


def test_source_authority_verifies_manifest_bytes_schema_size_and_rows(
    tmp_path: Path,
) -> None:
    shard = _organized_shard(tmp_path)

    source = resolve_symbol_day_source((shard,), "AAA", "2024-01-02")

    assert source.manifest_hash_verified is True
    assert (
        source.manifest_file_sha256
        == hashlib.sha256(shard.manifest_csv.read_bytes()).hexdigest()
    )
    assert source.manifest_size_verified is True
    assert source.manifest_rows_verified is True
    assert source.observed_sha256 == source.manifest_sha256
    assert source.qualifies_for_staging is True


def test_blank_legacy_hash_is_observed_but_not_historically_verified(
    tmp_path: Path,
) -> None:
    source = resolve_symbol_day_source(
        (_organized_shard(tmp_path, manifest_sha=False),), "AAA", "2024-01-02"
    )

    assert source.manifest_sha256 is None
    assert source.manifest_hash_verified is False
    assert len(source.observed_sha256) == 64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sha256", "f" * 64, "manifest SHA differs"),
        ("output_size", 1, "manifest size differs"),
        ("rows", 99, "manifest rows differ"),
        ("status", "failed", "does not qualify"),
    ],
)
def test_manifest_mismatch_or_failed_status_rejects_source(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    shard = _organized_shard(tmp_path)
    rows = list(csv.DictReader(shard.manifest_csv.open(newline="")))
    rows[0][field] = str(value)
    _write_manifest(shard.manifest_csv, rows)

    with pytest.raises(PolygonPITAlphaConversionError, match=message):
        resolve_symbol_day_source((shard,), "AAA", "2024-01-02")


def test_duplicate_manifest_rows_are_ambiguous(tmp_path: Path) -> None:
    shard = _organized_shard(tmp_path)
    rows = list(csv.DictReader(shard.manifest_csv.open(newline="")))
    rows.append(dict(rows[0]))
    _write_manifest(shard.manifest_csv, rows)

    with pytest.raises(PolygonPITAlphaConversionError, match="found 2"):
        resolve_symbol_day_source((shard,), "AAA", "2024-01-02")


def test_symlinked_manifest_source_is_rejected(tmp_path: Path) -> None:
    shard = _organized_shard(tmp_path)
    canonical = shard.second_aggs_root / "AAA" / "2024" / "01" / "2024-01-02.parquet"
    target = tmp_path / "elsewhere.parquet"
    canonical.replace(target)
    canonical.symlink_to(target)

    with pytest.raises(PolygonPITAlphaConversionError, match="non-symlink"):
        resolve_symbol_day_source((shard,), "AAA", "2024-01-02")


def test_audit_splits_staging_inventory_training_and_reportability(
    tmp_path: Path,
) -> None:
    audit = audit_organized_polygon_for_pit_alpha(
        (
            _organized_shard(
                tmp_path,
                manifest_sha=False,
                stale_path=True,
                extra_covariate=True,
                identity_transition=True,
            ),
        ),
        verify_canonical_files=True,
    )

    assert audit.staging_conversion_possible is True
    assert audit.bar_source_inventory_verified is False
    assert audit.pit_alpha_training_ready is False
    assert audit.reportable_pit_authority_ready is False
    assert audit.unexpected_covariate_symbols == ("CCC",)
    assert audit.source_symbol_identity_transitions == ("AAA",)
    assert "bar_source_inventory_not_verified" in audit.blockers
    assert "source_manifests_are_not_byte_authoritative" in audit.blockers
    shard = audit.shards[0]
    assert shard.blank_sha256_rows == 1
    assert shard.stale_output_path_rows == 1
    assert shard.observed_source_hash_rows == 1
    assert shard.manifest_hash_verified_rows == 0


def test_byte_verified_bar_inventory_still_does_not_authorize_training(
    tmp_path: Path,
) -> None:
    audit = audit_organized_polygon_for_pit_alpha(
        (_organized_shard(tmp_path),), verify_canonical_files=True
    )

    assert audit.staging_conversion_possible is True
    assert audit.bar_source_inventory_verified is True
    assert audit.pit_alpha_training_ready is False
    assert audit.reportable_pit_authority_ready is False
    assert "future_selected_universe" in audit.blockers


def test_audit_detects_hash_mismatch_and_unexpected_empty_file(
    tmp_path: Path,
) -> None:
    shard = _organized_shard(tmp_path)
    rows = list(csv.DictReader(shard.manifest_csv.open(newline="")))
    rows[0]["sha256"] = "f" * 64
    _write_manifest(shard.manifest_csv, rows)
    empty_path = shard.second_aggs_root / "BBB" / "2024" / "01" / "2024-01-02.parquet"
    empty_path.parent.mkdir(parents=True)
    _second_frame(symbol="BBB").to_parquet(empty_path, index=False)

    audit = audit_organized_polygon_for_pit_alpha((shard,), verify_canonical_files=True)

    observed = audit.shards[0]
    assert observed.canonical_hash_mismatch_rows == 1
    assert observed.unexpected_file_for_empty_rows == 1
    assert audit.bar_source_inventory_verified is False


def test_corporate_action_candidates_keep_full_date_and_file_provenance(
    tmp_path: Path,
) -> None:
    shard = _organized_shard(tmp_path)
    path = shard.covariates_root / "AAA" / "dividends.jsonl"

    candidate = next(
        iter_corporate_action_candidates(shard.covariates_root, symbols=("AAA",))
    )

    assert candidate.declaration_date == "2024-02-01"
    assert candidate.ex_date == "2024-03-01"
    assert candidate.record_date == "2024-03-04"
    assert candidate.payment_date == "2024-03-15"
    assert candidate.source_line_number == 1
    assert candidate.source_file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(candidate.source_record_sha256) == 64


def test_session_authority_round_trip_is_exact_and_create_only(tmp_path: Path) -> None:
    authority = _session(close_hour=13, special_reason="scheduled early close")
    path = tmp_path / "session.json"

    file_sha = write_exchange_session_authority(path, authority)

    assert (
        load_exchange_session_authority(path, expected_file_sha256=file_sha)
        == authority
    )
    with pytest.raises(PolygonPITAlphaConversionError, match="already exists"):
        write_exchange_session_authority(path, authority)


def test_conversion_publishes_exact_nonreportable_transaction(tmp_path: Path) -> None:
    shard = _organized_shard(tmp_path)
    source = resolve_symbol_day_source((shard,), "AAA", "2024-01-02")
    output = tmp_path / "staging" / "AAA.parquet"

    publication = convert_symbol_day_to_five_minute_staging(source, _session(), output)
    loaded = load_five_minute_staging_publication(
        output, expected_commit_file_sha256=publication["commit_file_sha256"]
    )

    assert output.is_file()
    assert publication["reportable_pit_authority"] is False
    assert publication["pit_alpha_training_ready"] is False
    assert loaded["receipt"]["missing_intervals_are_not_zero_filled"] is True
    assert (
        loaded["receipt"]["source_authority"]["source_receipt_sha256"]
        == source.source_receipt_sha256
    )
    assert loaded["receipt"]["scheduled_interval_count"] == 78
    assert len(loaded["receipt"]["output_table_semantic_sha256"]) == 64
    with pytest.raises(PolygonPITAlphaConversionError, match="already exists"):
        convert_symbol_day_to_five_minute_staging(source, _session(), output)


def test_repeated_conversion_has_stable_physical_and_semantic_hashes(
    tmp_path: Path,
) -> None:
    shard = _organized_shard(tmp_path)
    source = resolve_symbol_day_source((shard,), "AAA", "2024-01-02")
    one = tmp_path / "one" / "AAA.parquet"
    two = tmp_path / "two" / "AAA.parquet"

    first = convert_symbol_day_to_five_minute_staging(source, _session(), one)
    second = convert_symbol_day_to_five_minute_staging(source, _session(), two)
    first_loaded = load_five_minute_staging_publication(
        one, expected_commit_file_sha256=first["commit_file_sha256"]
    )
    second_loaded = load_five_minute_staging_publication(
        two, expected_commit_file_sha256=second["commit_file_sha256"]
    )

    assert first["output_file_sha256"] == second["output_file_sha256"]
    assert (
        first_loaded["receipt"]["output_table_semantic_sha256"]
        == second_loaded["receipt"]["output_table_semantic_sha256"]
    )


def test_source_mutation_after_authority_is_rejected(tmp_path: Path) -> None:
    shard = _organized_shard(tmp_path)
    source = resolve_symbol_day_source((shard,), "AAA", "2024-01-02")
    _second_frame().iloc[:1].to_parquet(Path(source.canonical_path), index=False)

    with pytest.raises(PolygonPITAlphaConversionError, match="drifted|changed"):
        convert_symbol_day_to_five_minute_staging(
            source, _session(), tmp_path / "staging" / "AAA.parquet"
        )


def test_manifest_mutation_after_authority_is_rejected(tmp_path: Path) -> None:
    shard = _organized_shard(tmp_path)
    source = resolve_symbol_day_source((shard,), "AAA", "2024-01-02")
    shard.manifest_csv.write_bytes(shard.manifest_csv.read_bytes() + b"\n")

    with pytest.raises(PolygonPITAlphaConversionError, match="SHA drifted"):
        convert_symbol_day_to_five_minute_staging(
            source, _session(), tmp_path / "staging" / "AAA.parquet"
        )


def test_mutated_receipt_invalidates_committed_publication(tmp_path: Path) -> None:
    shard = _organized_shard(tmp_path)
    source = resolve_symbol_day_source((shard,), "AAA", "2024-01-02")
    output = tmp_path / "staging" / "AAA.parquet"
    published = convert_symbol_day_to_five_minute_staging(source, _session(), output)
    receipt_path = output.with_suffix(".parquet.receipt.json")
    receipt_path.write_bytes(receipt_path.read_bytes() + b" ")

    with pytest.raises(PolygonPITAlphaConversionError, match="SHA drifted"):
        load_five_minute_staging_publication(
            output, expected_commit_file_sha256=published["commit_file_sha256"]
        )


def test_self_consistent_false_semantic_hash_is_independently_rejected(
    tmp_path: Path,
) -> None:
    shard = _organized_shard(tmp_path)
    source = resolve_symbol_day_source((shard,), "AAA", "2024-01-02")
    output = tmp_path / "staging" / "AAA.parquet"
    convert_symbol_day_to_five_minute_staging(source, _session(), output)
    receipt_path = output.with_suffix(".parquet.receipt.json")
    commit_path = output.with_suffix(".parquet.commit.json")

    receipt = json.loads(receipt_path.read_bytes())
    receipt["output_table_semantic_sha256"] = "f" * 64
    receipt["receipt_sha256"] = semantic_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    receipt_raw = canonical_json_file_bytes(receipt)
    receipt_path.write_bytes(receipt_raw)
    commit = json.loads(commit_path.read_bytes())
    commit["output_table_semantic_sha256"] = "f" * 64
    commit["receipt_file_sha256"] = hashlib.sha256(receipt_raw).hexdigest()
    commit["staging_receipt_sha256"] = receipt["receipt_sha256"]
    commit["commit_receipt_sha256"] = semantic_sha256(
        {key: value for key, value in commit.items() if key != "commit_receipt_sha256"}
    )
    commit_raw = canonical_json_file_bytes(commit)
    commit_path.write_bytes(commit_raw)

    with pytest.raises(PolygonPITAlphaConversionError, match="semantic receipt"):
        load_five_minute_staging_publication(
            output,
            expected_commit_file_sha256=hashlib.sha256(commit_raw).hexdigest(),
        )


def test_publication_link_failure_rolls_back_all_final_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = _organized_shard(tmp_path)
    source = resolve_symbol_day_source((shard,), "AAA", "2024-01-02")
    output = tmp_path / "staging" / "AAA.parquet"
    original = polygon._link_no_replace
    calls = 0

    def fail_second(source_path: Path, target_path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        original(source_path, target_path)

    monkeypatch.setattr(polygon, "_link_no_replace", fail_second)

    with pytest.raises(OSError, match="injected"):
        convert_symbol_day_to_five_minute_staging(source, _session(), output)
    assert not output.exists()
    assert not output.with_suffix(".parquet.receipt.json").exists()
    assert not output.with_suffix(".parquet.commit.json").exists()


def test_receipt_staging_failure_leaves_no_final_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = _organized_shard(tmp_path)
    source = resolve_symbol_day_source((shard,), "AAA", "2024-01-02")
    output = tmp_path / "staging" / "AAA.parquet"

    def fail_receipt(*_args: object, **_kwargs: object) -> Path:
        raise OSError("injected receipt staging failure")

    monkeypatch.setattr(polygon, "_write_temporary_bytes", fail_receipt)

    with pytest.raises(OSError, match="receipt staging"):
        convert_symbol_day_to_five_minute_staging(source, _session(), output)
    assert not output.exists()
    assert not output.with_suffix(".parquet.receipt.json").exists()
    assert not output.with_suffix(".parquet.commit.json").exists()


def test_tampered_source_authority_receipt_is_rejected(tmp_path: Path) -> None:
    source = resolve_symbol_day_source(
        (_organized_shard(tmp_path),), "AAA", "2024-01-02"
    )

    with pytest.raises(PolygonPITAlphaConversionError, match="receipt drifted"):
        replace(source, source_receipt_sha256="f" * 64).validate()
