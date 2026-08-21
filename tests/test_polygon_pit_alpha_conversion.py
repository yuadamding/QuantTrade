from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from rl_quant.data_sources.polygon_pit_alpha import (
    EXPECTED_NORMAL_SESSION_INTERVALS,
    OrganizedPolygonShard,
    PolygonPITAlphaConversionError,
    aggregate_polygon_second_bars_to_five_minutes,
    audit_organized_polygon_for_pit_alpha,
    convert_symbol_day_to_five_minute_staging,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256

EASTERN = ZoneInfo("America/New_York")


def _timestamp_ms(day: str, hour: int, minute: int, second: int) -> int:
    value = datetime.fromisoformat(
        f"{day}T{hour:02d}:{minute:02d}:{second:02d}"
    ).replace(tzinfo=EASTERN)
    return int(value.timestamp() * 1_000)


def _second_frame() -> pd.DataFrame:
    day = "2024-01-02"
    rows = [
        (9, 30, 1, 10.0, 10.5, 9.9, 10.2, 100.0, 10.1, 2),
        (9, 34, 59, 10.2, 11.0, 10.1, 10.9, 50.0, 10.8, 3),
        (9, 45, 0, 12.0, 12.5, 11.5, 12.2, 40.0, 12.1, 1),
    ]
    return pd.DataFrame(
        {
            "symbol": ["AAA"] * len(rows),
            "timestamp_ms": [_timestamp_ms(day, *row[:3]) for row in rows],
            "open": [row[3] for row in rows],
            "high": [row[4] for row in rows],
            "low": [row[5] for row in rows],
            "close": [row[6] for row in rows],
            "volume": [row[7] for row in rows],
            "vwap": [row[8] for row in rows],
            "transactions": [row[9] for row in rows],
            "adjusted": [True] * len(rows),
        }
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _organized_shard(tmp_path: Path) -> OrganizedPolygonShard:
    bars = tmp_path / "bars"
    covariates = tmp_path / "covariates"
    bars.mkdir()
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
    canonical.write_bytes(b"source")
    with (bars / "manifest.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "symbol",
                "date",
                "status",
                "rows",
                "output_path",
                "output_size",
                "sha256",
                "elapsed_seconds",
                "error",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "symbol": "AAA",
                "date": "2024-01-02",
                "status": "downloaded",
                "rows": 1,
                "output_path": "old-root/AAA/2024/01/2024-01-02.parquet",
                "sha256": "",
            }
        )
        writer.writerow(
            {
                "symbol": "BBB",
                "date": "2024-01-02",
                "status": "empty",
                "rows": 0,
                "output_path": "bars/BBB/2024/01/2024-01-02.parquet",
                "sha256": "",
            }
        )
    for symbol in ("AAA", "BBB", "CCC"):
        (covariates / symbol).mkdir()
    _write_jsonl(
        covariates / "AAA" / "overview_snapshots.jsonl",
        [
            {
                "asof_date": "2024-01-01",
                "ticker": "AAA",
                "share_class_figi": "FIGI-1",
                "composite_figi": "COMP-1",
                "cik": "1",
            },
            {
                "asof_date": "2024-02-01",
                "ticker": "AAA",
                "share_class_figi": "FIGI-2",
                "composite_figi": "COMP-2",
                "cik": "2",
            },
        ],
    )
    _write_jsonl(
        covariates / "BBB" / "overview_snapshots.jsonl",
        [
            {
                "asof_date": "2024-01-01",
                "ticker": "BBB",
                "share_class_figi": "FIGI-1",
                "composite_figi": "COMP-1",
                "cik": "1",
            }
        ],
    )
    _write_jsonl(
        covariates / "AAA" / "dividends.jsonl",
        [
            {
                "id": "div-1",
                "ex_dividend_date": "2024-03-01",
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
    result = aggregate_polygon_second_bars_to_five_minutes(_second_frame())

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
    assert first["available_at_ms"] == _timestamp_ms("2024-01-02", 9, 35, 0)
    assert result.iloc[1]["available_at_ms"] == _timestamp_ms("2024-01-02", 9, 50, 0)


def test_conversion_is_no_clobber_and_marks_output_nonreportable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "staging" / "bars.parquet"
    _second_frame().to_parquet(source, index=False)

    receipt = convert_symbol_day_to_five_minute_staging(source, output)

    assert output.is_file()
    assert receipt["reportable_pit_authority"] is False
    assert receipt["missing_intervals_are_not_zero_filled"] is True
    assert receipt["observed_interval_count"] == 2
    assert (
        receipt["output_file_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    )
    assert receipt["receipt_sha256"] == semantic_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    with pytest.raises(PolygonPITAlphaConversionError, match="already exists"):
        convert_symbol_day_to_five_minute_staging(source, output)


def test_audit_reports_conversion_inputs_but_refuses_pit_authority(
    tmp_path: Path,
) -> None:
    audit = audit_organized_polygon_for_pit_alpha(
        (_organized_shard(tmp_path),),
        verify_canonical_files=True,
    )

    assert audit.expected_universe_symbol_count == 2
    assert audit.observed_covariate_symbol_count == 3
    assert audit.unexpected_covariate_symbols == ("CCC",)
    assert audit.source_symbol_identity_transitions == ("AAA",)
    assert audit.cross_ticker_share_class_figis == ("FIGI-1",)
    assert audit.dividend_candidate_count == 1
    assert audit.corporate_actions_missing_announcement_count == 1
    assert audit.development_five_minute_conversion_ready is True
    assert audit.reportable_pit_authority_ready is False
    assert "future_selected_universe" in audit.blockers
    assert "unresolved_permanent_security_identity_transitions" in audit.blockers
    assert "source_manifests_are_not_byte_authoritative" in audit.blockers
    shard = audit.shards[0]
    assert shard.blank_sha256_rows == 2
    assert shard.stale_output_path_rows == 1
    assert shard.canonical_missing_file_rows == 0
