from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

from rl_quant.alpha.pit_universe import (
    ListingEventRecord,
    PITSecurityUniverseAuthority,
    PITUniverseRuleSpec,
    SourcedSecurityMasterRecord,
    SourcedTickerHistoryRecord,
    UniverseRankInputRecord,
)
from rl_quant.data_sources.massive.economic_provider_capture_v7 import (
    MassiveEconomicProviderCaptureV7Error,
    capture_massive_economic_rest_surface_for_test_v7,
    capture_massive_economic_rest_surface_v7,
    parse_massive_economic_raw_rest_capture_v7,
)
from rl_quant.features.massive_economic_authority_v7 import (
    MASSIVE_CASH_ACCRUAL_CONVENTION_V7_RECEIPT_SHA256,
    MASSIVE_ECONOMIC_AUTHORITY_V7_HISTORICAL_PANEL_AUTHORIZED,
    MASSIVE_ECONOMIC_AUTHORITY_V7_PREDICTIVE_TRAINING_AUTHORIZED,
    MASSIVE_ECONOMIC_AUTHORITY_V7_PROFITABILITY_REPORTING_AUTHORIZED,
    MassiveCashAccrualPeriodV7,
    MassiveCashLotV7,
    MassiveEconomicAuthorityV7Error,
    MassiveProviderEconomicOrderSnapshotV7,
    accrue_massive_cash_lots_v7,
    adapt_massive_raw_economic_captures_at_origin_v7,
    materialize_massive_resolved_economic_authority_at_origin_v7,
    parse_massive_resolved_economic_authority_at_origin_v7,
    select_massive_order_snapshot_at_origin_v7,
)
from rl_quant.protocol.canonical_artifact import semantic_sha256

DIGEST = "a" * 64
DECISION_AT_MS = 1_720_000_000_000


def _identity() -> PITSecurityUniverseAuthority:
    rule = PITUniverseRuleSpec.build(
        rule_id="economic-provider-v7-test",
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
            delisting_at_ms=None,
            successor_security_id=None,
            corporate_action_chain_id=f"CHAIN-{security_id}",
            identity_source_receipt_sha256=semantic_sha256((security_id, "master")),
        )
        for security_id in ("SEC-A", "SEC-B")
    )
    tickers = tuple(
        SourcedTickerHistoryRecord(
            security_id=security_id,
            ticker=ticker,
            valid_from_ms=1,
            valid_to_ms=None,
            available_at_ms=1,
            primary_exchange="XNYS",
            source_receipt_sha256=semantic_sha256((security_id, "ticker")),
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
            source_receipt_sha256=semantic_sha256((security_id, "listing")),
        )
        for security_id, ticker in (("SEC-A", "AAA"), ("SEC-B", "BBB"))
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
            source_receipt_sha256=semantic_sha256((security_id, "rank")),
        )
        for security_id in ("SEC-A", "SEC-B")
    )
    return PITSecurityUniverseAuthority.build(
        rule=rule,
        security_master=masters,
        ticker_history=tickers,
        listing_events=listings,
        delisting_events=(),
        rank_inputs=ranks,
    )


def _dividend_row(*, event_id: str = "DIV-1", ticker: str = "AAA") -> dict[str, object]:
    return {
        "id": event_id,
        "ticker": ticker,
        "cash_amount": 1.25,
        "currency": "USD",
        "declaration_date": "2024-05-01",
        "dividend_type": "CD",
        "ex_dividend_date": "2024-05-20",
        "frequency": 4,
        "pay_date": "2024-06-01",
        "record_date": "2024-05-21",
    }


def _body(rows: list[object], *, next_url: str | None = None) -> bytes:
    payload: dict[str, object] = {
        "status": "OK",
        "results": rows,
        "count": len(rows),
    }
    if next_url is not None:
        payload["next_url"] = next_url
    return json.dumps(payload, separators=(",", ":")).encode()


def _capture(
    root: Path,
    *,
    bodies: list[bytes],
    completed_at_ms: int,
    capture_id: str,
):
    root.mkdir(parents=True, exist_ok=True)
    return capture_massive_economic_rest_surface_for_test_v7(
        root=root,
        surface_id="massive-dividends",
        query_parameters={"ticker": "AAA"},
        raw_page_bodies=bodies,
        requested_at_ms=completed_at_ms - 100,
        completed_at_ms=completed_at_ms,
        entitlement_receipt_sha256=DIGEST,
        capture_id=capture_id,
    )


def _period(*, start: int = 100, end: int = 200, value: float = 0.10):
    return MassiveCashAccrualPeriodV7.build(
        period_id=f"CASH-{start}-{end}",
        period_start_at_ms=start,
        period_end_at_ms=end,
        one_period_return=value,
        available_at_ms=end,
        provider_id="qualified-cash-provider",
        provider_dataset="overnight-cash-total-return",
        raw_provider_source_receipt_sha256=semantic_sha256("cash-source"),
        raw_provider_row_receipt_sha256=semantic_sha256((start, end, value)),
        availability_rule_receipt_sha256=semantic_sha256("cash-publication-time"),
    )


def _snapshot(
    *,
    domain: str,
    keys: tuple[str, ...],
    ordered: tuple[str, ...],
    available: int,
    snapshot_id: str,
    effective_at_ms: int = 100,
):
    provisional = MassiveProviderEconomicOrderSnapshotV7(
        order_group_id="GROUP-1",
        order_snapshot_id=snapshot_id,
        interaction_domain_sha256=domain,
        effective_at_ms=effective_at_ms,
        snapshot_available_at_ms=available,
        complete_logical_event_keys=tuple(sorted(keys)),
        ordered_logical_event_keys=ordered,
        raw_provider_source_receipt_sha256=semantic_sha256(
            ("order-source", snapshot_id)
        ),
        raw_provider_snapshot_receipt_sha256=semantic_sha256(
            ("order-row", snapshot_id)
        ),
        provider_runtime_qualified=False,
        receipt_sha256="0" * 64,
    )
    result = replace(
        provisional, receipt_sha256=semantic_sha256(provisional.unsigned())
    )
    result.validate()
    return result


def test_raw_capture_preserves_exact_bytes_and_is_nonauthorizing(tmp_path: Path):
    raw = b'{ "status": "OK", "results": [], "count": 0 }\n'
    captured = _capture(
        tmp_path,
        bodies=[raw],
        completed_at_ms=DECISION_AT_MS,
        capture_id="raw",
    )

    assert captured.provider_runtime_qualified is False
    assert captured.pages[0].raw_body() == raw
    assert "apikey" not in captured.pages[0].request_url.lower()
    assert (
        parse_massive_economic_raw_rest_capture_v7(
            root=tmp_path, loaded_source=captured.loaded_source
        )
        == captured
    )

    tampered = replace(captured.pages[0], raw_response_body_base64="AAAA")
    with pytest.raises(MassiveEconomicProviderCaptureV7Error):
        tampered.validate(page_count=1)

    production_parameters = inspect.signature(
        capture_massive_economic_rest_surface_v7
    ).parameters
    assert "now_ms" not in production_parameters
    assert "provider_client" not in production_parameters
    assert "raw_page_bodies" not in production_parameters


def test_native_split_uses_exact_massive_execution_date_and_ratio(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = _body(
        [
            {
                "id": "SPLIT-1",
                "ticker": "AAA",
                "execution_date": "2024-05-20",
                "split_from": 2.0,
                "split_to": 1.0,
            }
        ]
    )
    captured = capture_massive_economic_rest_surface_for_test_v7(
        root=tmp_path,
        surface_id="massive-splits",
        query_parameters={"ticker": "AAA"},
        raw_page_bodies=(raw,),
        requested_at_ms=DECISION_AT_MS - 100,
        completed_at_ms=DECISION_AT_MS,
        entitlement_receipt_sha256=DIGEST,
        capture_id="split",
    )

    events, _ = adapt_massive_raw_economic_captures_at_origin_v7(
        captures=(captured,),
        identity_authority=_identity(),
        decision_at_ms=DECISION_AT_MS,
    )
    assert len(events) == 1
    assert events[0].kind == "reverse-split"
    assert events[0].share_ratio == 0.5
    assert events[0].availability_derivation_kind == (
        "conservative-response-completion"
    )


def test_raw_capture_rejects_non_ok_provider_status(tmp_path: Path):
    with pytest.raises(MassiveEconomicProviderCaptureV7Error):
        _capture(
            tmp_path,
            bodies=[
                json.dumps({"status": "ERROR", "results": [], "count": 0}).encode()
            ],
            completed_at_ms=DECISION_AT_MS,
            capture_id="error",
        )


def test_cash_accrues_only_lots_held_before_period_start():
    period = _period()
    assert (
        period.convention_receipt_sha256
        == MASSIVE_CASH_ACCRUAL_CONVENTION_V7_RECEIPT_SHA256
    )
    lots = (
        MassiveCashLotV7.build(
            lot_id="OPENING",
            amount=100.0,
            acquired_at_ms=99,
            acquisition_event_key="OPENING-CASH",
        ),
        MassiveCashLotV7.build(
            lot_id="DIVIDEND",
            amount=10.0,
            acquired_at_ms=100,
            acquisition_event_key="DIVIDEND-AT-START",
        ),
        MassiveCashLotV7.build(
            lot_id="MERGER",
            amount=20.0,
            acquired_at_ms=150,
            acquisition_event_key="MERGER-INTRAPERIOD",
        ),
        MassiveCashLotV7.build(
            lot_id="TENDER",
            amount=30.0,
            acquired_at_ms=200,
            acquisition_event_key="TENDER-AT-END",
        ),
    )

    accrued = accrue_massive_cash_lots_v7(cash_lots=lots, period=period)
    amounts = {row.lot_id: row.amount for row in accrued}
    assert amounts == {
        "DIVIDEND": 10.0,
        "MERGER": 20.0,
        "OPENING": pytest.approx(110.0),
        "TENDER": 30.0,
    }
    by_id = {row.lot_id: row for row in accrued}
    assert by_id["OPENING"].applied_accrual_period_receipts == (period.receipt_sha256,)
    assert by_id["DIVIDEND"].applied_accrual_period_receipts == ()
    with pytest.raises(MassiveEconomicAuthorityV7Error):
        accrue_massive_cash_lots_v7(cash_lots=accrued, period=period)


def test_order_selection_uses_one_latest_complete_snapshot():
    domain = semantic_sha256("domain")
    keys = (semantic_sha256("A"), semantic_sha256("B"))
    old = _snapshot(
        domain=domain,
        keys=keys,
        ordered=keys,
        available=1_000,
        snapshot_id="OLD",
    )
    new = _snapshot(
        domain=domain,
        keys=keys,
        ordered=tuple(reversed(keys)),
        available=2_000,
        snapshot_id="NEW",
    )
    conflict = _snapshot(
        domain=domain,
        keys=keys,
        ordered=keys,
        available=2_000,
        snapshot_id="CONFLICT",
    )

    assert (
        select_massive_order_snapshot_at_origin_v7(
            interaction_domain_sha256=domain,
            effective_at_ms=100,
            logical_event_keys=keys,
            snapshots=(old, new),
            decision_at_ms=2_000,
        )
        == new
    )
    with pytest.raises(MassiveEconomicAuthorityV7Error):
        select_massive_order_snapshot_at_origin_v7(
            interaction_domain_sha256=domain,
            effective_at_ms=100,
            logical_event_keys=keys,
            snapshots=(old, new, conflict),
            decision_at_ms=2_000,
        )


def test_future_malformed_page_changes_audit_not_origin_semantics(tmp_path: Path):
    identity = _identity()
    next_url = "https://api.massive.com/v3/reference/dividends?cursor=NEXT"
    current_only = _capture(
        tmp_path / "current",
        bodies=[_body([_dividend_row()])],
        completed_at_ms=DECISION_AT_MS,
        capture_id="current",
    )
    with_future = _capture(
        tmp_path / "future",
        bodies=[
            _body([_dividend_row()], next_url=next_url),
            _body([{"id": "BROKEN-FUTURE"}]),
        ],
        completed_at_ms=DECISION_AT_MS + 1,
        capture_id="future",
    )

    current_rows, current_audit = adapt_massive_raw_economic_captures_at_origin_v7(
        captures=(current_only,),
        identity_authority=identity,
        decision_at_ms=DECISION_AT_MS,
    )
    future_rows, future_audit = adapt_massive_raw_economic_captures_at_origin_v7(
        captures=(with_future,),
        identity_authority=identity,
        decision_at_ms=DECISION_AT_MS,
    )

    assert current_rows == future_rows
    assert current_audit.receipt_sha256 != future_audit.receipt_sha256
    assert len(future_audit.issue_inventory) == 1

    current_result = materialize_massive_resolved_economic_authority_at_origin_v7(
        root=tmp_path / "current",
        capture_loaded_sources=(current_only.loaded_source,),
        identity_authority=identity,
        decision_at_ms=DECISION_AT_MS,
        cash_accrual_periods=(),
        selected_order_snapshots=(),
        entitlement_receipt_sha256=DIGEST,
        artifact_id="semantic",
        committed_at_ms=DECISION_AT_MS + 10,
    )
    future_result = materialize_massive_resolved_economic_authority_at_origin_v7(
        root=tmp_path / "future",
        capture_loaded_sources=(with_future.loaded_source,),
        identity_authority=identity,
        decision_at_ms=DECISION_AT_MS,
        cash_accrual_periods=(),
        selected_order_snapshots=(),
        entitlement_receipt_sha256=DIGEST,
        artifact_id="semantic",
        committed_at_ms=DECISION_AT_MS + 10,
    )
    assert current_result.semantic_receipt_sha256 == (
        future_result.semantic_receipt_sha256
    )
    assert current_result.audit_receipt_sha256 != future_result.audit_receipt_sha256


def test_origin_authority_round_trip_has_distinct_semantic_and_audit_receipts(
    tmp_path: Path,
):
    identity = _identity()
    captured = _capture(
        tmp_path,
        bodies=[_body([_dividend_row()])],
        completed_at_ms=DECISION_AT_MS,
        capture_id="origin",
    )
    result = materialize_massive_resolved_economic_authority_at_origin_v7(
        root=tmp_path,
        capture_loaded_sources=(captured.loaded_source,),
        identity_authority=identity,
        decision_at_ms=DECISION_AT_MS,
        cash_accrual_periods=(),
        selected_order_snapshots=(),
        entitlement_receipt_sha256=DIGEST,
        artifact_id="origin",
        committed_at_ms=DECISION_AT_MS + 10,
    )

    assert len(result.selected_native_events) == 1
    assert result.provider_runtime_qualified is False
    assert result.semantic_receipt_sha256 != result.audit_receipt_sha256
    assert (
        parse_massive_resolved_economic_authority_at_origin_v7(
            root=tmp_path, loaded_source=result.loaded_source
        )
        == result
    )

    with pytest.raises(MassiveEconomicAuthorityV7Error):
        replace(result, audit_receipt_sha256=DIGEST).validate()


def test_materializer_requires_exact_complete_snapshot_for_a_real_tie(
    tmp_path: Path,
):
    identity = _identity()
    captured = _capture(
        tmp_path,
        bodies=[
            _body(
                [
                    _dividend_row(event_id="DIV-A"),
                    _dividend_row(event_id="DIV-B"),
                ]
            )
        ],
        completed_at_ms=DECISION_AT_MS,
        capture_id="tie",
    )
    events, _ = adapt_massive_raw_economic_captures_at_origin_v7(
        captures=(captured,),
        identity_authority=identity,
        decision_at_ms=DECISION_AT_MS,
    )
    keys = tuple(row.logical_event_key for row in events)
    domain = semantic_sha256(("massive-native-economic-interaction-v7", "CHAIN-SEC-A"))

    with pytest.raises(MassiveEconomicAuthorityV7Error):
        materialize_massive_resolved_economic_authority_at_origin_v7(
            root=tmp_path,
            capture_loaded_sources=(captured.loaded_source,),
            identity_authority=identity,
            decision_at_ms=DECISION_AT_MS,
            cash_accrual_periods=(),
            selected_order_snapshots=(),
            entitlement_receipt_sha256=DIGEST,
            artifact_id="missing-snapshot",
            committed_at_ms=DECISION_AT_MS + 10,
        )

    snapshot = _snapshot(
        domain=domain,
        keys=keys,
        ordered=tuple(reversed(keys)),
        available=DECISION_AT_MS,
        snapshot_id="COMPLETE",
        effective_at_ms=events[0].effective_at_ms,
    )
    result = materialize_massive_resolved_economic_authority_at_origin_v7(
        root=tmp_path,
        capture_loaded_sources=(captured.loaded_source,),
        identity_authority=identity,
        decision_at_ms=DECISION_AT_MS,
        cash_accrual_periods=(),
        selected_order_snapshots=(snapshot,),
        entitlement_receipt_sha256=DIGEST,
        artifact_id="complete-snapshot",
        committed_at_ms=DECISION_AT_MS + 10,
    )
    assert result.selected_order_snapshot_receipts == (snapshot.receipt_sha256,)


def test_v7_keeps_profitability_authorizations_false():
    assert MASSIVE_ECONOMIC_AUTHORITY_V7_HISTORICAL_PANEL_AUTHORIZED is False
    assert MASSIVE_ECONOMIC_AUTHORITY_V7_PREDICTIVE_TRAINING_AUTHORIZED is False
    assert MASSIVE_ECONOMIC_AUTHORITY_V7_PROFITABILITY_REPORTING_AUTHORIZED is False
