from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import rl_quant.data_sources.polygon_pit_alpha as polygon_staging
from rl_quant.alpha import (
    DelistingEventRecord,
    ListingEventRecord,
    PITAlphaDataError,
    PITSecurityUniverseAuthority,
    PITUniverseRuleSpec,
    PolygonStagingInventoryRecord,
    SourcedSecurityMasterRecord,
    SourcedTickerHistoryRecord,
    UniverseRankInputRecord,
    audit_polygon_staging_coverage,
    build_historical_membership,
    load_pit_security_universe,
    materialize_pit_security_universe,
)


def _rule() -> PITUniverseRuleSpec:
    return PITUniverseRuleSpec.build(
        rule_id="pit-test-trailing-dollar-volume-v1",
        target_size=2,
        ranking_lookback_sessions=3,
        ranking_lag_sessions=1,
        minimum_observed_sessions=2,
        minimum_close_price=1.0,
        minimum_average_dollar_volume=0.0,
        rebalance_frequency="monthly",
    )


def _masters() -> tuple[SourcedSecurityMasterRecord, ...]:
    return (
        SourcedSecurityMasterRecord(
            "SEC-A",
            "ISS-A",
            "XNYS",
            "COMMON",
            "common-stock",
            100,
            None,
            None,
            None,
            "a" * 64,
        ),
        SourcedSecurityMasterRecord(
            "SEC-B",
            "ISS-B",
            "XNAS",
            "COMMON",
            "common-stock",
            100,
            1_500,
            None,
            "CHAIN-B",
            "b" * 64,
        ),
        SourcedSecurityMasterRecord(
            "SEC-C",
            "ISS-C",
            "XNYS",
            "COMMON",
            "common-stock",
            100,
            None,
            None,
            None,
            "c" * 64,
        ),
    )


def _tickers() -> tuple[SourcedTickerHistoryRecord, ...]:
    return (
        SourcedTickerHistoryRecord("SEC-A", "AAA", 100, None, 90, "XNYS", "d" * 64),
        SourcedTickerHistoryRecord("SEC-B", "BBB", 100, 1_500, 90, "XNAS", "e" * 64),
        SourcedTickerHistoryRecord("SEC-C", "CCC", 100, None, 90, "XNYS", "f" * 64),
    )


def _listings() -> tuple[ListingEventRecord, ...]:
    return (
        ListingEventRecord("LIST-A", "SEC-A", 100, 90, "XNYS", "AAA", "1" * 64),
        ListingEventRecord("LIST-B", "SEC-B", 100, 90, "XNAS", "BBB", "2" * 64),
        ListingEventRecord("LIST-C", "SEC-C", 100, 90, "XNYS", "CCC", "3" * 64),
    )


def _delistings() -> tuple[DelistingEventRecord, ...]:
    return (
        DelistingEventRecord(
            "DELIST-B",
            "SEC-B",
            1_500,
            1_400,
            "exchange-delisting",
            None,
            "4" * 64,
        ),
    )


def _rank(
    security_id: str,
    *,
    effective_at_ms: int,
    effective_session_index: int,
    average_dollar_volume: float,
    close_price: float = 10.0,
) -> UniverseRankInputRecord:
    end_index = effective_session_index - 1
    start_index = end_index - 2
    return UniverseRankInputRecord(
        security_id=security_id,
        effective_at_ms=effective_at_ms,
        effective_session_index=effective_session_index,
        available_at_ms=effective_at_ms - 50,
        observation_start_ms=effective_at_ms - 300,
        observation_end_ms=effective_at_ms - 100,
        observation_start_session_index=start_index,
        observation_end_session_index=end_index,
        observed_session_count=3,
        average_dollar_volume=average_dollar_volume,
        close_price=close_price,
        source_receipt_sha256=hashlib.sha256(
            f"{security_id}:{effective_at_ms}".encode()
        ).hexdigest(),
    )


def _rank_inputs() -> tuple[UniverseRankInputRecord, ...]:
    return (
        _rank(
            "SEC-A",
            effective_at_ms=1_000,
            effective_session_index=10,
            average_dollar_volume=100.0,
        ),
        _rank(
            "SEC-B",
            effective_at_ms=1_000,
            effective_session_index=10,
            average_dollar_volume=90.0,
        ),
        _rank(
            "SEC-C",
            effective_at_ms=1_000,
            effective_session_index=10,
            average_dollar_volume=80.0,
        ),
        _rank(
            "SEC-A",
            effective_at_ms=2_000,
            effective_session_index=20,
            average_dollar_volume=70.0,
        ),
        _rank(
            "SEC-C",
            effective_at_ms=2_000,
            effective_session_index=20,
            average_dollar_volume=110.0,
        ),
    )


def _authority() -> PITSecurityUniverseAuthority:
    return PITSecurityUniverseAuthority.build(
        rule=_rule(),
        security_master=_masters(),
        ticker_history=_tickers(),
        listing_events=_listings(),
        delisting_events=_delistings(),
        rank_inputs=_rank_inputs(),
    )


def _inventory(symbol: str, session_date: str) -> PolygonStagingInventoryRecord:
    return PolygonStagingInventoryRecord(
        symbol=symbol,
        session_date=session_date,
        source_receipt_sha256=hashlib.sha256(
            f"source:{symbol}:{session_date}".encode()
        ).hexdigest(),
        output_file_sha256=hashlib.sha256(
            f"output:{symbol}:{session_date}".encode()
        ).hexdigest(),
        commit_receipt_sha256=hashlib.sha256(
            f"commit:{symbol}:{session_date}".encode()
        ).hexdigest(),
    )


def test_future_delisting_does_not_remove_current_pit_member() -> None:
    authority = _authority()

    first = {
        row.security_id: row
        for row in authority.membership_events
        if row.effective_at_ms == 1_000
    }
    second = {
        row.security_id: row
        for row in authority.membership_events
        if row.effective_at_ms == 2_000
    }

    assert first["SEC-A"].is_member
    assert first["SEC-A"].universe_rank == 1
    assert first["SEC-B"].is_member
    assert first["SEC-B"].universe_rank == 2
    assert not first["SEC-C"].is_member
    assert second["SEC-C"].is_member
    assert second["SEC-C"].universe_rank == 1
    assert not second["SEC-B"].is_member
    assert second["SEC-B"].eligibility_reason == "delisted"
    authority.validate()


def test_rank_inputs_must_end_by_t_minus_one() -> None:
    bad = replace(
        _rank_inputs()[0],
        observation_end_session_index=10,
        observation_start_session_index=8,
    )

    with pytest.raises(PITAlphaDataError, match="t or future"):
        bad.validate_for(_rule())


def test_every_active_security_requires_a_rank_input() -> None:
    with pytest.raises(PITAlphaDataError, match="complete active candidate set"):
        build_historical_membership(
            rule=_rule(),
            security_master=_masters(),
            listing_events=_listings(),
            delisting_events=_delistings(),
            rank_inputs=tuple(
                row
                for row in _rank_inputs()
                if not (row.security_id == "SEC-C" and row.effective_at_ms == 1_000)
            ),
        )


def test_tie_breaking_is_permanent_security_id_ascending() -> None:
    inputs = tuple(
        replace(row, average_dollar_volume=100.0)
        if row.effective_at_ms == 1_000
        else row
        for row in _rank_inputs()
    )
    rows = build_historical_membership(
        rule=_rule(),
        security_master=_masters(),
        listing_events=_listings(),
        delisting_events=_delistings(),
        rank_inputs=inputs,
    )
    first = {row.security_id: row for row in rows if row.effective_at_ms == 1_000}

    assert first["SEC-A"].universe_rank == 1
    assert first["SEC-B"].universe_rank == 2
    assert first["SEC-C"].universe_rank == 3


def test_membership_mutation_fails_independent_reconstruction() -> None:
    authority = _authority()
    mutated = replace(
        authority,
        membership_events=(
            replace(authority.membership_events[0], is_member=False),
            *authority.membership_events[1:],
        ),
    )

    with pytest.raises(PITAlphaDataError, match="independent PIT rank"):
        mutated.validate()


def test_identity_graph_rejects_overlapping_ticker_reuse() -> None:
    tickers = (
        _tickers()[0],
        replace(_tickers()[1], ticker="AAA", primary_exchange="XNYS"),
        _tickers()[2],
    )
    masters = (
        _masters()[0],
        replace(_masters()[1], primary_exchange="XNYS"),
        _masters()[2],
    )
    listings = (
        _listings()[0],
        replace(_listings()[1], ticker="AAA", primary_exchange="XNYS"),
        _listings()[2],
    )

    with pytest.raises(PITAlphaDataError, match="ticker history intervals overlap"):
        PITSecurityUniverseAuthority.build(
            rule=_rule(),
            security_master=masters,
            ticker_history=tickers,
            listing_events=listings,
            delisting_events=_delistings(),
            rank_inputs=_rank_inputs(),
        )


def test_identity_graph_rejects_successor_cycles() -> None:
    masters = (
        replace(_masters()[0], successor_security_id="SEC-C"),
        _masters()[1],
        replace(_masters()[2], successor_security_id="SEC-A"),
    )

    with pytest.raises(PITAlphaDataError, match="successor graph contains a cycle"):
        PITSecurityUniverseAuthority.build(
            rule=_rule(),
            security_master=masters,
            ticker_history=_tickers(),
            listing_events=_listings(),
            delisting_events=_delistings(),
            rank_inputs=_rank_inputs(),
        )


def test_identity_graph_rejects_gaps_in_one_security_ticker_history() -> None:
    first = replace(_tickers()[0], valid_to_ms=500)
    second = SourcedTickerHistoryRecord(
        "SEC-A",
        "AAA",
        600,
        None,
        590,
        "XNYS",
        "9" * 64,
    )

    with pytest.raises(PITAlphaDataError, match="ticker history has a gap"):
        PITSecurityUniverseAuthority.build(
            rule=_rule(),
            security_master=_masters(),
            ticker_history=(first, second, *_tickers()[1:]),
            listing_events=_listings(),
            delisting_events=_delistings(),
            rank_inputs=_rank_inputs(),
        )


def test_materializer_writes_required_files_but_never_authorizes_training(
    tmp_path: Path,
) -> None:
    publication = materialize_pit_security_universe(tmp_path, _authority())

    expected = {
        "security_master.parquet",
        "ticker_history.parquet",
        "listing_events.parquet",
        "delisting_events.parquet",
        "membership_events.parquet",
        "universe_rank_inputs.parquet",
        "universe_rule.json",
        "identity_universe_authority.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    assert publication["identity_authority_ready"] is True
    assert publication["historical_universe_authority_ready"] is True
    assert publication["pit_alpha_training_ready"] is False
    assert publication["reportable_pit_authority_ready"] is False
    security = pd.read_parquet(tmp_path / "security_master.parquet")
    assert list(security["security_id"]) == ["SEC-A", "SEC-B", "SEC-C"]
    stored = json.loads((tmp_path / "identity_universe_authority.json").read_bytes())
    for row in stored["files"]:
        path = tmp_path / row["relative_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["file_sha256"]

    with pytest.raises(PITAlphaDataError, match="must be empty"):
        materialize_pit_security_universe(tmp_path, _authority())


def test_bundle_loader_reopens_exact_files_and_reconstructs_membership(
    tmp_path: Path,
) -> None:
    authority = _authority()
    publication = materialize_pit_security_universe(tmp_path, authority)

    loaded = load_pit_security_universe(
        tmp_path,
        expected_authority_file_sha256=publication["authority_file_sha256"],
    )

    assert loaded == authority


def test_bundle_loader_rejects_file_mutation(tmp_path: Path) -> None:
    publication = materialize_pit_security_universe(tmp_path, _authority())
    membership = tmp_path / "membership_events.parquet"
    membership.write_bytes(membership.read_bytes() + b"drift")

    with pytest.raises(PITAlphaDataError, match="SHA-256 drifted"):
        load_pit_security_universe(
            tmp_path,
            expected_authority_file_sha256=publication["authority_file_sha256"],
        )


def test_materialization_is_physically_deterministic(tmp_path: Path) -> None:
    one = tmp_path / "one"
    two = tmp_path / "two"

    materialize_pit_security_universe(one, _authority())
    materialize_pit_security_universe(two, _authority())

    for name in (
        "security_master.parquet",
        "ticker_history.parquet",
        "listing_events.parquet",
        "delisting_events.parquet",
        "membership_events.parquet",
        "universe_rank_inputs.parquet",
        "universe_rule.json",
    ):
        assert (one / name).read_bytes() == (two / name).read_bytes()


def test_polygon_coverage_identifies_missing_delisted_and_historical_names() -> None:
    audit = audit_polygon_staging_coverage(
        authority=_authority(),
        required_sessions=(
            ("2024-01-02", 1_000),
            ("2024-02-01", 2_000),
        ),
        inventory=(
            _inventory("AAA", "2024-01-02"),
            _inventory("BBB", "2024-01-02"),
            _inventory("AAA", "2024-02-01"),
            _inventory("ZZZ", "2024-01-02"),
        ),
    )

    assert audit.required_symbol_day_count == 5
    assert audit.covered_symbol_day_count == 3
    assert audit.missing_symbol_days == (
        "SEC-C|CCC|2024-01-02",
        "SEC-C|CCC|2024-02-01",
    )
    assert audit.unused_inventory_symbol_days == ("ZZZ|2024-01-02",)
    assert audit.coverage_fraction == pytest.approx(0.6)
    assert not audit.bar_source_inventory_complete
    assert not audit.pit_alpha_training_ready
    assert not audit.reportable_pit_authority_ready


def test_complete_polygon_coverage_still_cannot_authorize_training() -> None:
    audit = audit_polygon_staging_coverage(
        authority=_authority(),
        required_sessions=(("2024-01-02", 1_000),),
        inventory=tuple(
            _inventory(symbol, "2024-01-02") for symbol in ("AAA", "BBB", "CCC")
        ),
    )

    assert audit.bar_source_inventory_complete
    assert audit.coverage_fraction == 1.0
    assert not audit.pit_alpha_training_ready
    assert not audit.reportable_pit_authority_ready


@pytest.mark.parametrize(
    ("symbol", "session_date", "message"),
    (("../AAA", "2024-01-02", "ticker"), ("AAA", "2024-13-01", "ISO")),
)
def test_polygon_inventory_rejects_unsafe_keys(
    symbol: str, session_date: str, message: str
) -> None:
    with pytest.raises(PITAlphaDataError, match=message):
        PolygonStagingInventoryRecord(
            symbol,
            session_date,
            "a" * 64,
            "b" * 64,
            "c" * 64,
        ).validate()


def test_committed_polygon_bundles_feed_coverage_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def load_publication(
        output_path: Path, *, expected_commit_file_sha256: str
    ) -> dict[str, object]:
        assert expected_commit_file_sha256 == "a" * 64
        symbol = output_path.stem
        return {
            "receipt": {
                "symbol": symbol,
                "session_date": "2024-01-02",
                "source_authority": {"source_receipt_sha256": "b" * 64},
            },
            "commit": {"commit_receipt_sha256": "c" * 64},
            "output_file_sha256": "d" * 64,
        }

    monkeypatch.setattr(
        polygon_staging, "load_five_minute_staging_publication", load_publication
    )

    observed = polygon_staging.load_polygon_staging_inventory(
        (
            (tmp_path / "BBB.parquet", "a" * 64),
            (tmp_path / "AAA.parquet", "a" * 64),
        )
    )

    assert tuple(row.symbol for row in observed) == ("AAA", "BBB")
    assert all(row.source_receipt_sha256 == "b" * 64 for row in observed)
