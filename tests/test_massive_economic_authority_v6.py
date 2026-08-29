from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_quant.alpha.contracts import CorporateActionKind, TerminalEventKind
from rl_quant.alpha.pit_universe import (
    ListingEventRecord,
    PITSecurityUniverseAuthority,
    PITUniverseRuleSpec,
    SourcedSecurityMasterRecord,
    SourcedTickerHistoryRecord,
    UniverseRankInputRecord,
)
from rl_quant.data_sources.massive.source_receipts import (
    LoadedMassiveSourceObject,
    load_massive_source_bundle,
    publish_massive_source_object,
)
from rl_quant.evaluation.massive_adaptive_economic_event_transition_v1 import (
    MassiveAdaptiveEconomicEventTransitionV1Error,
    apply_massive_adaptive_postfill_events_v1,
    apply_massive_adaptive_prefill_events_v1,
    build_massive_adaptive_economic_event_transition_v1,
)
from rl_quant.features.massive_adaptive_fill_source_v1 import adaptive_fill_clock_v1
from rl_quant.features.massive_economic_authority_v6 import (
    MASSIVE_ECONOMIC_AUTHORITY_V6_HISTORICAL_PANEL_AUTHORIZED,
    MASSIVE_ECONOMIC_AUTHORITY_V6_PREDICTIVE_TRAINING_AUTHORIZED,
    MASSIVE_ECONOMIC_AUTHORITY_V6_PROFITABILITY_REPORTING_AUTHORIZED,
    MASSIVE_ECONOMIC_AUTHORITY_V6_SPEC_SHA256,
    MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_DATASETS,
    MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_KINDS,
    MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_OBJECT_PREFIX,
    MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_SCHEMA,
    MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_SOURCE_SCHEMA_SHA256,
    MassiveEconomicAuthorityV6Error,
    build_massive_provider_economic_archive_authority_v6,
    capture_massive_raw_provider_economic_source_v6,
    parse_massive_raw_provider_economic_source_v6,
    resolve_massive_economic_authority_at_origin_v6,
)
from rl_quant.protocol.canonical_artifact import (
    canonical_json_file_bytes,
    semantic_sha256,
)


DIGEST = "a" * 64
PROVIDER = "massive"
OBSERVED_AT_MS = 10_000


def _identity() -> PITSecurityUniverseAuthority:
    rule = PITUniverseRuleSpec.build(
        rule_id="economic-provider-v6-test",
        target_size=3,
        ranking_lookback_sessions=3,
        ranking_lag_sessions=1,
        minimum_observed_sessions=2,
        minimum_close_price=1.0,
        minimum_average_dollar_volume=0.0,
        rebalance_frequency="monthly",
    )
    chains = {"SEC-A": "CHAIN-AB", "SEC-B": "CHAIN-AB", "SEC-C": "CHAIN-C"}
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
            corporate_action_chain_id=chain,
            identity_source_receipt_sha256=semantic_sha256((security_id, "master")),
        )
        for security_id, chain in chains.items()
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
        for security_id, ticker in (
            ("SEC-A", "AAA"),
            ("SEC-B", "BBB"),
            ("SEC-C", "CCC"),
        )
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
        for security_id, ticker in (
            ("SEC-A", "AAA"),
            ("SEC-B", "BBB"),
            ("SEC-C", "CCC"),
        )
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
        for security_id in chains
    )
    return PITSecurityUniverseAuthority.build(
        rule=rule,
        security_master=masters,
        ticker_history=tickers,
        listing_events=listings,
        delisting_events=(),
        rank_inputs=ranks,
    )


def _corporate(
    *,
    provider_event_key: str,
    provider_revision_id: str,
    security_id: str,
    effective_at_ms: int,
    available_at_ms: int,
    cash_per_share: float,
    supersedes: str | None = None,
    status: str = "active",
    successor_security_id: str | None = None,
    successor_ratio: float = 0.0,
    kind: CorporateActionKind | None = None,
    share_ratio: float = 1.0,
) -> dict[str, object]:
    resolved_kind = kind or (
        CorporateActionKind.SPIN_OFF
        if successor_security_id is not None
        else CorporateActionKind.CASH_DIVIDEND
    )
    effective_field = {
        CorporateActionKind.CASH_DIVIDEND: "ex_dividend_at_ms",
        CorporateActionKind.SPIN_OFF: "entitlement_at_ms",
        CorporateActionKind.SPLIT: "execution_at_ms",
    }[resolved_kind]
    return {
        "provider_request_id": "request-corporate",
        "provider_event_key": provider_event_key,
        "provider_revision_id": provider_revision_id,
        "supersedes_provider_revision_id": supersedes,
        "revision_status": status,
        "provider_available_at_ms": available_at_ms,
        "provider_row_locator": f"corporate/{provider_event_key}/{provider_revision_id}",
        "provider_record": {
            "kind": resolved_kind.value,
            "security_id": security_id,
            effective_field: effective_at_ms,
            "cash_per_share": cash_per_share,
            "share_ratio": share_ratio,
            "successor_security_id": successor_security_id,
            "successor_ratio": successor_ratio,
            "affected_fraction": 1.0,
        },
    }


def _order(
    *, provider_event_key: str, effective_at_ms: int, sequence: int, available: int
) -> dict[str, object]:
    return {
        "provider_request_id": "request-order",
        "event_provider_id": PROVIDER,
        "event_provider_dataset": "provider-corporate-actions",
        "event_source_kind": "corporate-actions",
        "provider_event_key": provider_event_key,
        "provider_order_available_at_ms": available,
        "provider_row_locator": f"order/{provider_event_key}/{available}",
        "provider_record": {
            "provider_effective_at_ms": effective_at_ms,
            "provider_local_economic_sequence": sequence,
        },
    }


def _terminal(
    *,
    provider_event_key: str,
    security_id: str,
    effective_at_ms: int,
    available_at_ms: int,
) -> dict[str, object]:
    return {
        "provider_request_id": "request-terminal",
        "provider_event_key": provider_event_key,
        "provider_revision_id": "r0",
        "supersedes_provider_revision_id": None,
        "revision_status": "active",
        "provider_available_at_ms": available_at_ms,
        "provider_row_locator": f"terminal/{provider_event_key}/r0",
        "provider_record": {
            "kind": TerminalEventKind.WORTHLESS.value,
            "security_id": security_id,
            "zero_value_effective_at_ms": effective_at_ms,
            "cash_per_share": 0.0,
            "successor_security_id": None,
            "successor_ratio": 0.0,
        },
    }


def _cash_return(*, effective_at_ms: int, available_at_ms: int) -> dict[str, object]:
    return {
        "provider_request_id": "request-cash",
        "provider_event_key": "CASH-RETURN-1",
        "provider_revision_id": "r0",
        "supersedes_provider_revision_id": None,
        "revision_status": "active",
        "provider_available_at_ms": available_at_ms,
        "provider_row_locator": "cash/CASH-RETURN-1/r0",
        "provider_record": {
            "kind": "cash-return",
            "accrual_period_end_at_ms": effective_at_ms,
            "one_step_return": 0.01,
        },
    }


def _publish(
    root: Path,
    *,
    source_kind: str,
    records: list[dict[str, object]],
    suffix: str,
) -> LoadedMassiveSourceObject:
    request_id = {
        "corporate-actions": "request-corporate",
        "terminal-outcomes": "request-terminal",
        "cash-returns": "request-cash",
        "economic-order-observations": "request-order",
    }[source_kind]
    availability = tuple(
        int(
            row.get(
                "provider_available_at_ms",
                row.get("provider_order_available_at_ms", 0),
            )
        )
        for row in records
    )
    observed_at_ms = max((OBSERVED_AT_MS, *availability))
    payload = {
        "schema": MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_SCHEMA,
        "source_kind": source_kind,
        "provider_id": PROVIDER,
        "provider_dataset": f"provider-{source_kind}",
        "provider_endpoint": "https://api.massive.example/reference",
        "query_start_at_ms": 0,
        "query_end_at_ms": 9_000,
        "provider_observed_at_ms": observed_at_ms,
        "provider_request_ids": [request_id],
        "pagination_complete": True,
        "page_count": 1,
        "records": records,
    }
    relative = (
        f"{MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_OBJECT_PREFIX}"
        f"{source_kind}-{suffix}.json"
    )
    publish_massive_source_object(
        stream=BytesIO(canonical_json_file_bytes(payload)),
        root=root,
        relative_payload_path=relative,
        dataset_id=MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_DATASETS[source_kind],
        source_object_key=relative,
        requested_at_ms=observed_at_ms - 1,
        downloaded_at_ms=observed_at_ms,
        schema_sha256=MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_SOURCE_SCHEMA_SHA256,
        entitlement_receipt_sha256=DIGEST,
        committed_at_ms=observed_at_ms,
        etag=semantic_sha256((source_kind, suffix, records)),
        request_id=request_id,
    )
    return load_massive_source_bundle(
        root=root,
        relative_payload_path=relative,
        verified_at_ms=observed_at_ms,
    )


def _archive(
    root: Path,
    *,
    corporate: list[dict[str, object]],
    terminal: list[dict[str, object]] | None = None,
    cash: list[dict[str, object]] | None = None,
    orders: list[dict[str, object]] | None = None,
    suffix: str = "base",
    reverse_rows: bool = False,
):
    root.mkdir(parents=True, exist_ok=True)
    identity = _identity()
    by_role = {
        "corporate-actions": list(corporate),
        "terminal-outcomes": list(terminal or []),
        "cash-returns": list(cash or []),
        "economic-order-observations": list(orders or []),
    }
    if reverse_rows:
        by_role = {role: list(reversed(rows)) for role, rows in by_role.items()}
    loaded = tuple(
        _publish(
            root,
            source_kind=role,
            records=by_role[role],
            suffix=suffix,
        )
        for role in MASSIVE_RAW_PROVIDER_ECONOMIC_SOURCE_V6_KINDS
    )
    return (
        identity,
        build_massive_provider_economic_archive_authority_v6(
            root=root,
            loaded_sources=loaded,
            identity_authority=identity,
        ),
        loaded,
    )


def test_v6_spec_keeps_performance_authorization_false() -> None:
    assert len(MASSIVE_ECONOMIC_AUTHORITY_V6_SPEC_SHA256) == 64
    assert not MASSIVE_ECONOMIC_AUTHORITY_V6_HISTORICAL_PANEL_AUTHORIZED
    assert not MASSIVE_ECONOMIC_AUTHORITY_V6_PREDICTIVE_TRAINING_AUTHORIZED
    assert not MASSIVE_ECONOMIC_AUTHORITY_V6_PROFITABILITY_REPORTING_AUTHORIZED


def test_raw_provider_bytes_derive_logical_and_effective_identity(
    tmp_path: Path,
) -> None:
    row = _corporate(
        provider_event_key="DIV-1",
        provider_revision_id="r0",
        security_id="SEC-A",
        effective_at_ms=2_000,
        available_at_ms=1_500,
        cash_per_share=1.0,
    )
    loaded = _publish(
        tmp_path,
        source_kind="corporate-actions",
        records=[row],
        suffix="parse",
    )
    source = parse_massive_raw_provider_economic_source_v6(
        root=tmp_path, loaded_source=loaded
    )
    observation = source.event_observations[0]
    assert observation.source_event.provider_event_key == "DIV-1"
    assert observation.source_event.logical_event_key == semantic_sha256(
        {
            "provider_id": PROVIDER,
            "provider_dataset": "provider-corporate-actions",
            "source_kind": "corporate-actions",
            "provider_event_key": "DIV-1",
        }
    )
    assert observation.source_event.event.effective_at_ms == 2_000
    assert observation.source_event.effective_timestamp_contract == (
        "ex-distribution-economic-time"
    )


def test_package_capture_exhausts_provider_pages(tmp_path: Path) -> None:
    first = _corporate(
        provider_event_key="DIV-A",
        provider_revision_id="r0",
        security_id="SEC-A",
        effective_at_ms=2_000,
        available_at_ms=1_000,
        cash_per_share=1.0,
    )
    second = _corporate(
        provider_event_key="DIV-C",
        provider_revision_id="r0",
        security_id="SEC-C",
        effective_at_ms=2_001,
        available_at_ms=1_000,
        cash_per_share=2.0,
    )
    first.pop("provider_request_id")
    second.pop("provider_request_id")

    class Client:
        def paginate_economic_observations(self, **_: object):
            return (
                {
                    "ResponseMetadata": {"RequestId": "request-page-1"},
                    "Records": [first],
                    "IsTruncated": True,
                    "NextToken": "next",
                },
                {
                    "ResponseMetadata": {"RequestId": "request-page-2"},
                    "Records": [second],
                    "IsTruncated": False,
                    "NextToken": None,
                },
            )

    ticks = iter((9_000, 10_000))
    captured = capture_massive_raw_provider_economic_source_v6(
        provider_client=Client(),
        root=tmp_path,
        source_kind="corporate-actions",
        provider_id=PROVIDER,
        provider_dataset="provider-corporate-actions",
        provider_endpoint="https://api.massive.example/reference",
        query_start_at_ms=0,
        query_end_at_ms=9_000,
        entitlement_receipt_sha256=DIGEST,
        capture_id="two-pages",
        now_ms=lambda: next(ticks),
    )
    assert captured.page_count == 2
    assert captured.provider_request_ids == ("request-page-1", "request-page-2")
    assert len(captured.event_observations) == 2
    assert captured.loaded_source.receipt.request_id == "request-page-2"


def test_package_capture_rejects_open_pagination(tmp_path: Path) -> None:
    class Client:
        def paginate_economic_observations(self, **_: object):
            return (
                {
                    "ResponseMetadata": {"RequestId": "request-page-1"},
                    "Records": [],
                    "IsTruncated": True,
                    "NextToken": "unconsumed",
                },
            )

    ticks = iter((9_000, 10_000))
    with pytest.raises(MassiveEconomicAuthorityV6Error, match="close exactly"):
        capture_massive_raw_provider_economic_source_v6(
            provider_client=Client(),
            root=tmp_path,
            source_kind="corporate-actions",
            provider_id=PROVIDER,
            provider_dataset="provider-corporate-actions",
            provider_endpoint="https://api.massive.example/reference",
            query_start_at_ms=0,
            query_end_at_ms=9_000,
            entitlement_receipt_sha256=DIGEST,
            capture_id="open-pagination",
            now_ms=lambda: next(ticks),
        )


def test_wrong_provider_effective_field_fails(tmp_path: Path) -> None:
    row = _corporate(
        provider_event_key="DIV-1",
        provider_revision_id="r0",
        security_id="SEC-A",
        effective_at_ms=2_000,
        available_at_ms=1_500,
        cash_per_share=1.0,
    )
    provider_record = dict(row["provider_record"])
    provider_record["payment_at_ms"] = provider_record.pop("ex_dividend_at_ms")
    row["provider_record"] = provider_record
    loaded = _publish(
        tmp_path,
        source_kind="corporate-actions",
        records=[row],
        suffix="bad-effective",
    )
    with pytest.raises(
        MassiveEconomicAuthorityV6Error,
        match="field inventory|effective-time field",
    ):
        parse_massive_raw_provider_economic_source_v6(
            root=tmp_path, loaded_source=loaded
        )


def test_revision_and_cancellation_are_origin_causal(tmp_path: Path) -> None:
    rows = [
        _corporate(
            provider_event_key="DIV-1",
            provider_revision_id="r0",
            security_id="SEC-A",
            effective_at_ms=2_000,
            available_at_ms=1_000,
            cash_per_share=1.0,
        ),
        _corporate(
            provider_event_key="DIV-1",
            provider_revision_id="r1",
            supersedes="r0",
            status="corrected",
            security_id="SEC-A",
            effective_at_ms=2_000,
            available_at_ms=3_000,
            cash_per_share=1.1,
        ),
        _corporate(
            provider_event_key="DIV-1",
            provider_revision_id="r2",
            supersedes="r1",
            status="cancelled",
            security_id="SEC-A",
            effective_at_ms=2_000,
            available_at_ms=5_000,
            cash_per_share=1.1,
        ),
    ]
    identity, archive, _ = _archive(tmp_path, corporate=rows)
    early = resolve_massive_economic_authority_at_origin_v6(
        archive=archive, identity_authority=identity, decision_at_ms=2_500
    )
    corrected = resolve_massive_economic_authority_at_origin_v6(
        archive=archive, identity_authority=identity, decision_at_ms=4_000
    )
    cancelled = resolve_massive_economic_authority_at_origin_v6(
        archive=archive, identity_authority=identity, decision_at_ms=6_000
    )
    assert early.selected_events[0].source_event.event.cash_per_share == 1.0
    assert corrected.selected_events[0].source_event.event.cash_per_share == 1.1
    assert not cancelled.selected_events
    assert cancelled.cancelled_logical_event_keys == (
        early.selected_events[0].source_event.logical_event_key,
    )


def test_same_provider_key_cannot_form_two_root_events(tmp_path: Path) -> None:
    rows = [
        _corporate(
            provider_event_key="DIV-1",
            provider_revision_id="root-a",
            security_id="SEC-A",
            effective_at_ms=2_000,
            available_at_ms=1_000,
            cash_per_share=1.0,
        ),
        _corporate(
            provider_event_key="DIV-1",
            provider_revision_id="root-b",
            security_id="SEC-A",
            effective_at_ms=2_000,
            available_at_ms=1_100,
            cash_per_share=1.1,
        ),
    ]
    with pytest.raises(MassiveEconomicAuthorityV6Error, match="one active root"):
        _archive(tmp_path, corporate=rows)


def test_unrelated_interaction_domains_need_no_global_order(tmp_path: Path) -> None:
    rows = [
        _corporate(
            provider_event_key="DIV-A",
            provider_revision_id="r0",
            security_id="SEC-A",
            effective_at_ms=2_000,
            available_at_ms=1_000,
            cash_per_share=1.0,
        ),
        _corporate(
            provider_event_key="DIV-C",
            provider_revision_id="r0",
            security_id="SEC-C",
            effective_at_ms=2_000,
            available_at_ms=1_000,
            cash_per_share=2.0,
        ),
    ]
    identity, archive, _ = _archive(tmp_path, corporate=rows)
    resolved = resolve_massive_economic_authority_at_origin_v6(
        archive=archive, identity_authority=identity, decision_at_ms=3_000
    )
    assert len(resolved.selected_events) == 2
    assert {
        row.order_evidence.local_economic_sequence for row in resolved.selected_events
    } == {0}
    assert len({row.interaction_domain_sha256 for row in resolved.selected_events}) == 2


def test_same_interaction_tie_requires_origin_available_order(tmp_path: Path) -> None:
    rows = [
        _corporate(
            provider_event_key="DIV-A",
            provider_revision_id="r0",
            security_id="SEC-A",
            effective_at_ms=2_000,
            available_at_ms=1_000,
            cash_per_share=1.0,
        ),
        _corporate(
            provider_event_key="DIV-B",
            provider_revision_id="r0",
            security_id="SEC-B",
            effective_at_ms=2_000,
            available_at_ms=1_000,
            cash_per_share=2.0,
        ),
    ]
    orders = [
        _order(
            provider_event_key="DIV-A",
            effective_at_ms=2_000,
            sequence=0,
            available=4_000,
        ),
        _order(
            provider_event_key="DIV-B",
            effective_at_ms=2_000,
            sequence=1,
            available=4_000,
        ),
    ]
    identity, archive, _ = _archive(tmp_path, corporate=rows, orders=orders)
    with pytest.raises(MassiveEconomicAuthorityV6Error, match="origin-available"):
        resolve_massive_economic_authority_at_origin_v6(
            archive=archive, identity_authority=identity, decision_at_ms=3_000
        )
    resolved = resolve_massive_economic_authority_at_origin_v6(
        archive=archive, identity_authority=identity, decision_at_ms=5_000
    )
    assert tuple(
        row.order_evidence.local_economic_sequence for row in resolved.selected_events
    ) == (0, 1)


def test_future_rows_do_not_change_earlier_semantic_origin_receipt(
    tmp_path: Path,
) -> None:
    base = _corporate(
        provider_event_key="DIV-A",
        provider_revision_id="r0",
        security_id="SEC-A",
        effective_at_ms=2_000,
        available_at_ms=1_000,
        cash_per_share=1.0,
    )
    identity_a, archive_a, _ = _archive(tmp_path / "a", corporate=[base], suffix="a")
    future_revision = _corporate(
        provider_event_key="DIV-A",
        provider_revision_id="r1",
        supersedes="r0",
        status="corrected",
        security_id="SEC-A",
        effective_at_ms=2_000,
        available_at_ms=7_000,
        cash_per_share=1.2,
    )
    future_unrelated = _corporate(
        provider_event_key="DIV-C",
        provider_revision_id="r0",
        security_id="SEC-C",
        effective_at_ms=2_000,
        available_at_ms=7_000,
        cash_per_share=3.0,
    )
    identity_b, archive_b, _ = _archive(
        tmp_path / "b",
        corporate=[base, future_revision, future_unrelated],
        suffix="b",
    )
    resolved_a = resolve_massive_economic_authority_at_origin_v6(
        archive=archive_a, identity_authority=identity_a, decision_at_ms=5_000
    )
    resolved_b = resolve_massive_economic_authority_at_origin_v6(
        archive=archive_b, identity_authority=identity_b, decision_at_ms=5_000
    )
    assert archive_a.receipt_sha256 != archive_b.receipt_sha256
    assert resolved_a.receipt_sha256 == resolved_b.receipt_sha256
    assert resolved_a.selected_revision_inventory_sha256 == (
        resolved_b.selected_revision_inventory_sha256
    )
    assert resolved_a.archive_audit_receipt_sha256 != (
        resolved_b.archive_audit_receipt_sha256
    )


def test_raw_source_row_permutation_does_not_change_origin_semantics(
    tmp_path: Path,
) -> None:
    rows = [
        _corporate(
            provider_event_key=f"DIV-{security_id}",
            provider_revision_id="r0",
            security_id=security_id,
            effective_at_ms=2_000 + index,
            available_at_ms=1_000,
            cash_per_share=float(index + 1),
        )
        for index, security_id in enumerate(("SEC-A", "SEC-C"))
    ]
    identity_a, archive_a, _ = _archive(tmp_path / "a", corporate=rows, suffix="a")
    identity_b, archive_b, _ = _archive(
        tmp_path / "b", corporate=rows, suffix="b", reverse_rows=True
    )
    resolved_a = resolve_massive_economic_authority_at_origin_v6(
        archive=archive_a, identity_authority=identity_a, decision_at_ms=5_000
    )
    resolved_b = resolve_massive_economic_authority_at_origin_v6(
        archive=archive_b, identity_authority=identity_b, decision_at_ms=5_000
    )
    assert resolved_a.receipt_sha256 == resolved_b.receipt_sha256


def test_order_receipt_substitution_fails(tmp_path: Path) -> None:
    rows = [
        _corporate(
            provider_event_key="DIV-A",
            provider_revision_id="r0",
            security_id="SEC-A",
            effective_at_ms=2_000,
            available_at_ms=1_000,
            cash_per_share=1.0,
        ),
        _corporate(
            provider_event_key="DIV-B",
            provider_revision_id="r0",
            security_id="SEC-B",
            effective_at_ms=2_000,
            available_at_ms=1_000,
            cash_per_share=2.0,
        ),
    ]
    orders = [
        _order(
            provider_event_key="DIV-A",
            effective_at_ms=2_000,
            sequence=0,
            available=2_500,
        ),
        _order(
            provider_event_key="DIV-B",
            effective_at_ms=2_000,
            sequence=1,
            available=2_500,
        ),
    ]
    identity, archive, _ = _archive(tmp_path, corporate=rows, orders=orders)
    resolved = resolve_massive_economic_authority_at_origin_v6(
        archive=archive, identity_authority=identity, decision_at_ms=3_000
    )
    tampered_order = replace(
        resolved.selected_events[0].order_evidence,
        provider_order_observation_receipt_sha256="f" * 64,
    )
    tampered_event = replace(resolved.selected_events[0], order_evidence=tampered_order)
    tampered = replace(
        resolved,
        selected_events=(tampered_event, *resolved.selected_events[1:]),
    )
    with pytest.raises(MassiveEconomicAuthorityV6Error, match="order evidence receipt"):
        tampered.validate()


def test_adaptive_event_transition_repairs_split_dividend_terminal_and_cash(
    tmp_path: Path,
) -> None:
    fill_date = "2024-01-03"
    prior_date = "2024-01-02"
    fill_start, fill_end = adaptive_fill_clock_v1(fill_date)
    prior_close = fill_start - 18 * 60 * 60 * 1_000
    fill_close = fill_end + 6 * 60 * 60 * 1_000
    corporate = [
        _corporate(
            provider_event_key="SPLIT-A",
            provider_revision_id="r0",
            security_id="SEC-A",
            effective_at_ms=fill_start - 1_000,
            available_at_ms=fill_start - 2_000,
            cash_per_share=0.0,
            kind=CorporateActionKind.SPLIT,
            share_ratio=2.0,
        ),
        _corporate(
            provider_event_key="DIV-A",
            provider_revision_id="r0",
            security_id="SEC-A",
            effective_at_ms=fill_end + 1_000,
            available_at_ms=fill_end,
            cash_per_share=1.0,
        ),
    ]
    terminal = [
        _terminal(
            provider_event_key="WORTHLESS-C",
            security_id="SEC-C",
            effective_at_ms=fill_end + 2_000,
            available_at_ms=fill_end + 1_000,
        )
    ]
    cash = [
        _cash_return(
            effective_at_ms=fill_end + 3_000,
            available_at_ms=fill_end + 2_000,
        )
    ]
    identity, archive, _ = _archive(
        tmp_path,
        corporate=corporate,
        terminal=terminal,
        cash=cash,
        suffix="adaptive-transition",
    )
    daily = SimpleNamespace(
        validate=lambda: None,
        sessions=(
            SimpleNamespace(
                source_session_date=prior_date,
                regular_close_at_ms=prior_close,
            ),
            SimpleNamespace(
                source_session_date=fill_date,
                regular_close_at_ms=fill_close,
            ),
        ),
    )
    transition = build_massive_adaptive_economic_event_transition_v1(
        prior_session_date=prior_date,
        fill_session_date=fill_date,
        provider_archive=archive,
        daily_input_authority=daily,
        identity_authority=identity,
    )
    assert len(transition.prefill_events) == 1
    assert len(transition.postfill_events) == 3

    shares, cash_after_prefill, requested = (
        apply_massive_adaptive_prefill_events_v1(
            transition=transition,
            existing_shares={"SEC-A": 10.0, "SEC-C": 5.0},
            cash=100.0,
            requested_shares={"SEC-A": 5.0, "SEC-C": -5.0},
        )
    )
    assert shares == {"SEC-A": 20.0, "SEC-C": 5.0}
    assert cash_after_prefill == 100.0
    assert requested == {"SEC-A": 10.0, "SEC-C": -5.0}

    final_shares, final_cash = apply_massive_adaptive_postfill_events_v1(
        transition=transition,
        shares={"SEC-A": 30.0, "SEC-C": 5.0},
        cash=100.0,
    )
    assert final_shares == {"SEC-A": 30.0}
    assert final_cash == pytest.approx(131.3)

    with pytest.raises(
        MassiveAdaptiveEconomicEventTransitionV1Error,
        match="archive|authority|receipt",
    ):
        build_massive_adaptive_economic_event_transition_v1(
            prior_session_date=prior_date,
            fill_session_date=fill_date,
            provider_archive=replace(archive, receipt_sha256="f" * 64),
            daily_input_authority=daily,
            identity_authority=identity,
        )
