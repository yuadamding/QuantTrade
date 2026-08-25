from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

from rl_quant.alpha.contracts import TerminalEventKind
from rl_quant.alpha.pit_universe import (
    DelistingEventRecord,
    ListingEventRecord,
    PITSecurityUniverseAuthority,
    PITUniverseRuleSpec,
    SourcedSecurityMasterRecord,
    SourcedTickerHistoryRecord,
    UniverseRankInputRecord,
)
from rl_quant.data_sources.massive.economic_provider_capture_v8 import (
    MASSIVE_ECONOMIC_REST_SURFACES_V8,
    MassiveEconomicProviderCaptureV8Error,
    build_massive_economic_query_parameters_v8,
    capture_massive_economic_rest_surface_for_test_v8,
    capture_massive_economic_rest_surface_v8,
    parse_massive_economic_raw_rest_capture_v8,
)
from rl_quant.features.massive_economic_coverage_v8 import (
    MASSIVE_ECONOMIC_COVERAGE_V8_HISTORICAL_PANEL_AUTHORIZED,
    MASSIVE_ECONOMIC_COVERAGE_V8_PREDICTIVE_TRAINING_AUTHORIZED,
    MASSIVE_ECONOMIC_COVERAGE_V8_PROFITABILITY_REPORTING_AUTHORIZED,
    MASSIVE_ZERO_CASH_POLICY_V8,
    MassiveEconomicCoverageScopeV8,
    MassiveEconomicCoverageV8Error,
    MassiveTerminalDispositionV8,
    MassiveZeroCashPolicyV8,
    adapt_massive_current_economic_captures_at_origin_v8,
    materialize_massive_economic_origin_coverage_v8,
    parse_massive_economic_origin_coverage_v8,
    parse_massive_terminal_coverage_source_v8,
    publish_massive_terminal_coverage_source_for_test_v8,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256

DIGEST = "a" * 64
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"
DECISION_AT_MS = 1_720_000_000_000
DELISTING_AT_MS = 1_717_068_600_000


def _identity(*, with_delisting: bool = True) -> PITSecurityUniverseAuthority:
    rule = PITUniverseRuleSpec.build(
        rule_id="economic-coverage-v8-test",
        target_size=2,
        ranking_lookback_sessions=3,
        ranking_lag_sessions=1,
        minimum_observed_sessions=2,
        minimum_close_price=1.0,
        minimum_average_dollar_volume=0.0,
        rebalance_frequency="monthly",
    )
    masters = tuple(
        SourcedSecurityMasterRecord(
            security_id=security_id,
            issuer_id=f"ISS-{security_id}",
            primary_exchange="XNYS",
            share_class="COMMON",
            security_type="common-stock",
            listing_at_ms=1,
            delisting_at_ms=(
                DELISTING_AT_MS if with_delisting and security_id == "SEC-B" else None
            ),
            successor_security_id=None,
            corporate_action_chain_id=f"CHAIN-{security_id}",
            identity_source_receipt_sha256=semantic_sha256((security_id, "master-v8")),
        )
        for security_id in ("SEC-A", "SEC-B")
    )
    tickers = tuple(
        SourcedTickerHistoryRecord(
            security_id=security_id,
            ticker=ticker,
            valid_from_ms=1,
            valid_to_ms=(
                DELISTING_AT_MS if with_delisting and security_id == "SEC-B" else None
            ),
            available_at_ms=1,
            primary_exchange="XNYS",
            source_receipt_sha256=semantic_sha256((security_id, "ticker-v8")),
        )
        for security_id, ticker in (("SEC-A", "AAA"), ("SEC-B", "BBB"))
    )
    listings = tuple(
        ListingEventRecord(
            event_id=f"LIST-{security_id}",
            security_id=security_id,
            effective_at_ms=1,
            available_at_ms=1,
            primary_exchange="XNYS",
            ticker=ticker,
            source_receipt_sha256=semantic_sha256((security_id, "listing-v8")),
        )
        for security_id, ticker in (("SEC-A", "AAA"), ("SEC-B", "BBB"))
    )
    delistings = (
        (
            DelistingEventRecord(
                event_id="DELIST-SEC-B",
                security_id="SEC-B",
                effective_at_ms=DELISTING_AT_MS,
                available_at_ms=DELISTING_AT_MS,
                reason="worthless",
                successor_security_id=None,
                source_receipt_sha256=semantic_sha256("delisting-sec-b-v8"),
            ),
        )
        if with_delisting
        else ()
    )
    ranks = tuple(
        UniverseRankInputRecord(
            security_id=security_id,
            effective_at_ms=100,
            effective_session_index=10,
            available_at_ms=99,
            observation_start_ms=1,
            observation_end_ms=98,
            observation_start_session_index=7,
            observation_end_session_index=9,
            observed_session_count=3,
            average_dollar_volume=1_000_000.0,
            close_price=100.0,
            source_receipt_sha256=semantic_sha256((security_id, "rank-v8")),
        )
        for security_id in ("SEC-A", "SEC-B")
    )
    return PITSecurityUniverseAuthority.build(
        rule=rule,
        security_master=masters,
        ticker_history=tickers,
        listing_events=listings,
        delisting_events=delistings,
        rank_inputs=ranks,
    )


def _body(rows: list[object], *, next_url: str | None = None) -> bytes:
    payload: dict[str, object] = {
        "status": "OK",
        "results": rows,
        "count": len(rows),
    }
    if next_url is not None:
        payload["next_url"] = next_url
    return json.dumps(payload, separators=(",", ":")).encode()


def _dividend(
    *,
    event_id: str = "DIV-1",
    ticker: str = "AAA",
    currency: str = "USD",
    ex_date: str = "2024-05-20",
    distribution_type: str = "recurring",
) -> dict[str, object]:
    return {
        "cash_amount": 1.25,
        "currency": currency,
        "declaration_date": "2024-05-01",
        "distribution_type": distribution_type,
        "ex_dividend_date": ex_date,
        "frequency": 4,
        "historical_adjustment_factor": 0.99,
        "id": event_id,
        "pay_date": "2024-06-01",
        "record_date": "2024-05-21",
        "split_adjusted_cash_amount": 1.25,
        "ticker": ticker,
    }


def _split(
    *,
    event_id: str = "SPLIT-1",
    ticker: str = "AAA",
    execution_date: str = "2024-06-03",
    adjustment_type: str = "forward_split",
    split_from: float = 1.0,
    split_to: float = 2.0,
) -> dict[str, object]:
    return {
        "adjustment_type": adjustment_type,
        "execution_date": execution_date,
        "historical_adjustment_factor": 0.5,
        "id": event_id,
        "split_from": split_from,
        "split_to": split_to,
        "ticker": ticker,
    }


def _capture(
    root: Path,
    *,
    surface_id: str,
    bodies: list[bytes],
    completed_at_ms: int,
    capture_id: str,
):
    root.mkdir(parents=True, exist_ok=True)
    return capture_massive_economic_rest_surface_for_test_v8(
        root=root,
        surface_id=surface_id,
        coverage_start_date=START_DATE,
        coverage_end_date=END_DATE,
        raw_page_bodies=bodies,
        requested_at_ms=completed_at_ms - 100,
        completed_at_ms=completed_at_ms,
        entitlement_receipt_sha256=DIGEST,
        capture_id=capture_id,
    )


def _captures(root: Path, *, completed_at_ms: int = DECISION_AT_MS):
    return (
        _capture(
            root,
            surface_id="massive-dividends-v1",
            bodies=[_body([_dividend()])],
            completed_at_ms=completed_at_ms,
            capture_id="dividends",
        ),
        _capture(
            root,
            surface_id="massive-splits-v1",
            bodies=[_body([_split()])],
            completed_at_ms=completed_at_ms,
            capture_id="splits",
        ),
    )


def _terminal(
    root: Path,
    *,
    identity: PITSecurityUniverseAuthority,
    include_disposition: bool = True,
):
    delisting = identity.delisting_events[0] if identity.delisting_events else None
    rows = (
        (
            MassiveTerminalDispositionV8.build(
                security_id="SEC-B",
                delisting_event_id="DELIST-SEC-B",
                terminal_kind=TerminalEventKind.WORTHLESS.value,
                effective_at_ms=DELISTING_AT_MS,
                provider_available_at_ms=DELISTING_AT_MS + 1,
                cash_per_share=0.0,
                successor_security_id=None,
                successor_ratio=0.0,
                delisting_source_receipt_sha256=(
                    delisting.source_receipt_sha256 if delisting is not None else DIGEST
                ),
                raw_provider_source_receipt_sha256=semantic_sha256(
                    "terminal-source-v8"
                ),
                raw_provider_row_receipt_sha256=semantic_sha256("terminal-row-v8"),
            ),
        )
        if include_disposition and delisting is not None
        else ()
    )
    return publish_massive_terminal_coverage_source_for_test_v8(
        root=root,
        coverage_start_date=START_DATE,
        coverage_end_date=END_DATE,
        dispositions=rows,
        entitlement_receipt_sha256=DIGEST,
        source_id="terminal",
        committed_at_ms=DECISION_AT_MS,
    )


def test_v8_uses_current_fixed_all_market_surfaces(tmp_path: Path):
    assert MASSIVE_ECONOMIC_REST_SURFACES_V8 == {
        "massive-dividends-v1": "/stocks/v1/dividends",
        "massive-splits-v1": "/stocks/v1/splits",
    }
    parameters = build_massive_economic_query_parameters_v8(
        surface_id="massive-dividends-v1",
        coverage_start_date=START_DATE,
        coverage_end_date=END_DATE,
    )
    assert ("ticker", "AAA") not in parameters
    assert all(key.lower() != "cursor" for key, _ in parameters)
    assert ("limit", "5000") in parameters

    captured = _capture(
        tmp_path,
        surface_id="massive-dividends-v1",
        bodies=[_body([])],
        completed_at_ms=DECISION_AT_MS,
        capture_id="current",
    )
    assert "/stocks/v1/dividends?" in captured.initial_request_url
    assert captured.fixed_runtime_captured is False
    assert (
        parse_massive_economic_raw_rest_capture_v8(
            root=tmp_path, loaded_source=captured.loaded_source
        ).fixed_runtime_captured
        is False
    )
    parameters_api = inspect.signature(
        capture_massive_economic_rest_surface_v8
    ).parameters
    assert "query_parameters" not in parameters_api
    assert "provider_client" not in parameters_api
    assert "now_ms" not in parameters_api


def test_v8_query_rejects_narrow_or_cursor_tampering(tmp_path: Path):
    captured = _capture(
        tmp_path,
        surface_id="massive-dividends-v1",
        bodies=[_body([])],
        completed_at_ms=DECISION_AT_MS,
        capture_id="query",
    )
    tampered = replace(
        captured,
        query_parameters=tuple(sorted((*captured.query_parameters, ("ticker", "AAA")))),
        receipt_sha256="0" * 64,
    )
    tampered = replace(
        tampered,
        receipt_sha256=semantic_sha256(tampered.semantic_unsigned()),
    )
    with pytest.raises(MassiveEconomicProviderCaptureV8Error):
        tampered.validate()


def test_v8_accepts_cursor_value_containing_apikey_text(tmp_path: Path):
    next_url = "https://api.massive.com/stocks/v1/dividends?cursor=containsapikeytext"
    captured = _capture(
        tmp_path,
        surface_id="massive-dividends-v1",
        bodies=[_body([], next_url=next_url), _body([])],
        completed_at_ms=DECISION_AT_MS,
        capture_id="cursor-value",
    )
    assert captured.page_count == 2


def test_capture_level_cutoff_never_admits_a_page_prefix(tmp_path: Path):
    identity = _identity(with_delisting=False)
    current = _capture(
        tmp_path / "current",
        surface_id="massive-dividends-v1",
        bodies=[_body([_dividend()])],
        completed_at_ms=DECISION_AT_MS,
        capture_id="current",
    )
    next_url = "https://api.massive.com/stocks/v1/dividends?cursor=NEXT"
    spanning = _capture(
        tmp_path / "spanning",
        surface_id="massive-dividends-v1",
        bodies=[
            _body([_dividend()], next_url=next_url),
            _body([_dividend(event_id="DIV-2", ex_date="2024-05-21")]),
        ],
        completed_at_ms=DECISION_AT_MS + 1,
        capture_id="spanning",
    )

    current_rows, current_future = adapt_massive_current_economic_captures_at_origin_v8(
        captures=(current,),
        identity_authority=identity,
        decision_at_ms=DECISION_AT_MS,
        accounting_lane="strict-pit-capture",
    )
    spanning_rows, spanning_future = (
        adapt_massive_current_economic_captures_at_origin_v8(
            captures=(spanning,),
            identity_authority=identity,
            decision_at_ms=DECISION_AT_MS,
            accounting_lane="strict-pit-capture",
        )
    )
    assert len(current_rows) == 1
    assert current_future == ()
    assert spanning_rows == ()
    assert spanning_future == (spanning.receipt_sha256,)


def test_finalized_accounting_lane_is_explicit_and_never_predictive(tmp_path: Path):
    identity = _identity(with_delisting=False)
    captured = _capture(
        tmp_path,
        surface_id="massive-dividends-v1",
        bodies=[_body([_dividend(distribution_type="supplemental")])],
        completed_at_ms=DECISION_AT_MS + 10_000,
        capture_id="finalized",
    )
    rows, future = adapt_massive_current_economic_captures_at_origin_v8(
        captures=(captured,),
        identity_authority=identity,
        decision_at_ms=DECISION_AT_MS,
        accounting_lane="finalized-accounting-research",
    )
    assert future == ()
    assert rows[0].accounting_lane == "finalized-accounting-research"
    assert rows[0].predictive_feature_eligible is False
    assert rows[0].classification == "supplemental"


def test_non_usd_dividend_fails_closed_without_fx_authority(tmp_path: Path):
    captured = _capture(
        tmp_path,
        surface_id="massive-dividends-v1",
        bodies=[_body([_dividend(currency="CAD")])],
        completed_at_ms=DECISION_AT_MS,
        capture_id="cad",
    )
    with pytest.raises(MassiveEconomicCoverageV8Error, match="non-USD"):
        adapt_massive_current_economic_captures_at_origin_v8(
            captures=(captured,),
            identity_authority=_identity(with_delisting=False),
            decision_at_ms=DECISION_AT_MS,
            accounting_lane="strict-pit-capture",
        )


def test_terminal_coverage_is_exact_and_required(tmp_path: Path):
    identity = _identity()
    captures = _captures(tmp_path)
    empty_terminal = _terminal(tmp_path, identity=identity, include_disposition=False)
    scope = MassiveEconomicCoverageScopeV8.build(
        coverage_start_date=START_DATE, coverage_end_date=END_DATE
    )
    with pytest.raises(MassiveEconomicCoverageV8Error, match="delistings"):
        materialize_massive_economic_origin_coverage_v8(
            root=tmp_path,
            capture_objects=captures,
            identity_authority=identity,
            terminal_loaded_source=empty_terminal.loaded_source,
            scope=scope,
            decision_at_ms=DECISION_AT_MS,
            accounting_lane="finalized-accounting-research",
            entitlement_receipt_sha256=DIGEST,
            artifact_id="missing-terminal",
            committed_at_ms=DECISION_AT_MS + 20,
        )


def test_complete_origin_coverage_round_trip_remains_nonauthorizing(tmp_path: Path):
    identity = _identity()
    captures = _captures(tmp_path)
    terminal = _terminal(tmp_path, identity=identity)
    result = materialize_massive_economic_origin_coverage_v8(
        root=tmp_path,
        capture_objects=captures,
        identity_authority=identity,
        terminal_loaded_source=terminal.loaded_source,
        scope=MassiveEconomicCoverageScopeV8.build(
            coverage_start_date=START_DATE,
            coverage_end_date=END_DATE,
        ),
        decision_at_ms=DECISION_AT_MS,
        accounting_lane="finalized-accounting-research",
        entitlement_receipt_sha256=DIGEST,
        artifact_id="complete",
        committed_at_ms=DECISION_AT_MS + 20,
    )

    assert result.coverage_qualified is True
    assert result.transport_qualified is False
    assert result.economic_authority_qualified is False
    assert len(result.selected_events) == 2
    assert result.cash_policy.policy_id == MASSIVE_ZERO_CASH_POLICY_V8
    assert result.cash_policy.accrual_period_receipts == ()
    assert result.cash_policy.negative_cash_allowed is False
    assert (
        parse_massive_economic_origin_coverage_v8(
            root=tmp_path, loaded_source=result.loaded_source
        )
        == result
    )
    assert (
        parse_massive_terminal_coverage_source_v8(
            root=tmp_path, loaded_source=terminal.loaded_source
        ).fixed_runtime_captured
        is False
    )


def test_missing_surface_and_irrelevant_capture_cannot_qualify(tmp_path: Path):
    identity = _identity()
    captures = _captures(tmp_path)
    terminal = _terminal(tmp_path, identity=identity)
    scope = MassiveEconomicCoverageScopeV8.build(
        coverage_start_date=START_DATE, coverage_end_date=END_DATE
    )
    with pytest.raises(MassiveEconomicCoverageV8Error, match="exactly one"):
        materialize_massive_economic_origin_coverage_v8(
            root=tmp_path,
            capture_objects=(captures[0],),
            identity_authority=identity,
            terminal_loaded_source=terminal.loaded_source,
            scope=scope,
            decision_at_ms=DECISION_AT_MS,
            accounting_lane="finalized-accounting-research",
            entitlement_receipt_sha256=DIGEST,
            artifact_id="missing-split",
            committed_at_ms=DECISION_AT_MS + 20,
        )


def test_zero_cash_policy_is_the_only_v8_policy():
    policy = MassiveZeroCashPolicyV8.build()
    policy.validate()
    with pytest.raises(MassiveEconomicCoverageV8Error):
        replace(policy, negative_cash_allowed=True).validate()


def test_v8_keeps_all_performance_authorizations_false():
    assert MASSIVE_ECONOMIC_COVERAGE_V8_HISTORICAL_PANEL_AUTHORIZED is False
    assert MASSIVE_ECONOMIC_COVERAGE_V8_PREDICTIVE_TRAINING_AUTHORIZED is False
    assert MASSIVE_ECONOMIC_COVERAGE_V8_PROFITABILITY_REPORTING_AUTHORIZED is False
