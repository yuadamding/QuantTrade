from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rl_quant.alpha import (
    AvailabilityRecord,
    AvailabilitySnapshot,
    CashReturnRecord,
    CorporateActionKind,
    CorporateActionRecord,
    DatasetFileRecord,
    EconomicPosition,
    EconomicValuePoint,
    IndependentEconomicReconciliation,
    MembershipEvent,
    PITAlphaDataError,
    PITAlphaDatasetAuthority,
    PITAlphaDatasetManifest,
    SecurityMasterRecord,
    TerminalEventKind,
    TerminalEventRecord,
    TickerHistoryRecord,
    UniverseRule,
    apply_corporate_action,
    apply_cash_return,
    availability_at,
    compute_post_fill_total_return,
    evaluate_pit_alpha_data_gate,
    load_pit_alpha_manifest,
    mark_position,
    membership_at,
    resolve_ticker,
    validate_manifest_files,
    write_pit_alpha_manifest,
)
from rl_quant.protocol.canonical_artifact import canonical_json_file_bytes, semantic_sha256


def _rule() -> UniverseRule:
    return UniverseRule(
        rule_id="active-500-trailing-liquidity-v1",
        rule_sha256="a" * 64,
        membership_mode="point-in-time-events",
        ranking_lookback_sessions=63,
        ranking_lag_sessions=1,
    )


def _manifest(files: tuple[DatasetFileRecord, ...]) -> PITAlphaDatasetManifest:
    return PITAlphaDatasetManifest.build(
        dataset_id="pit-alpha-golden-v1",
        action_axis=("CASH", "SEC-A", "SEC-B"),
        universe_rule=_rule(),
        files=files,
        source_receipts=("b" * 64, "c" * 64),
    )


def _terminal_stock_merger() -> TerminalEventRecord:
    return TerminalEventRecord(
        event_id="terminal-a",
        security_id="SEC-A",
        kind=TerminalEventKind.MERGER_STOCK,
        effective_at_ms=100,
        available_at_ms=90,
        successor_security_id="SEC-B",
        successor_ratio=0.5,
    )


def _authority() -> PITAlphaDatasetAuthority:
    manifest = _manifest(
        (
            DatasetFileRecord("authorities/security.json", 2, "d" * 64, "application/json"),
        )
    )
    return PITAlphaDatasetAuthority(
        manifest=manifest,
        security_master=(
            SecurityMasterRecord(
                security_id="SEC-A",
                issuer_id="ISSUER-A",
                primary_exchange="XNYS",
                share_class="COMMON-A",
                security_type="common-stock",
                listing_at_ms=10,
                delisting_at_ms=100,
                successor_security_id="SEC-B",
                corporate_action_chain_id="CHAIN-A",
            ),
            SecurityMasterRecord(
                security_id="SEC-B",
                issuer_id="ISSUER-B",
                primary_exchange="XNAS",
                share_class="COMMON",
                security_type="common-stock",
                listing_at_ms=20,
            ),
        ),
        ticker_history=(
            TickerHistoryRecord("SEC-A", "OLD", 10, 50, 9),
            TickerHistoryRecord("SEC-A", "NEW", 50, 100, 45),
            TickerHistoryRecord("SEC-B", "NEXT", 20, None, 19),
        ),
        membership_events=(
            MembershipEvent("SEC-A", 20, 19, 18, True, 1),
            MembershipEvent("SEC-A", 95, 94, 93, False),
            MembershipEvent("SEC-B", 30, 29, 28, True, 2),
        ),
        availability=(
            AvailabilityRecord(
                "SEC-A",
                20,
                19,
                AvailabilitySnapshot(True, True, True, True, False),
                "ordinary-session",
            ),
            AvailabilityRecord(
                "SEC-A",
                100,
                90,
                AvailabilitySnapshot(True, False, False, True, True),
                "stock-merger",
            ),
            AvailabilityRecord(
                "SEC-B",
                30,
                29,
                AvailabilitySnapshot(True, True, True, True, False),
                "ordinary-session",
            ),
        ),
        corporate_actions=(
            CorporateActionRecord(
                event_id="split-a",
                security_id="SEC-A",
                kind=CorporateActionKind.SPLIT,
                effective_at_ms=40,
                available_at_ms=35,
                share_ratio=2.0,
            ),
        ),
        terminal_events=(_terminal_stock_merger(),),
        cash_returns=(CashReturnRecord(20, 19, 0.0001, "e" * 64),),
    )


def test_permanent_security_identity_survives_ticker_change() -> None:
    rows = _authority().ticker_history
    assert resolve_ticker(rows, security_id="SEC-A", effective_at_ms=49, knowledge_at_ms=49) == "OLD"
    assert resolve_ticker(rows, security_id="SEC-A", effective_at_ms=50, knowledge_at_ms=49) == "NEW"
    position = EconomicPosition.from_mapping({"SEC-A": 3.0})
    assert position.as_mapping() == {"SEC-A": 3.0}


def test_future_membership_and_availability_cannot_change_current_action() -> None:
    membership = (
        MembershipEvent("SEC-A", 10, 9, 8, True, 1),
        MembershipEvent("SEC-A", 20, 20, 19, False),
    )
    assert membership_at(
        membership,
        security_ids=("SEC-A",),
        effective_at_ms=15,
        knowledge_at_ms=15,
    ) == {"SEC-A": True}

    states = (
        AvailabilityRecord(
            "SEC-A",
            10,
            9,
            AvailabilitySnapshot(True, True, True, True, False),
            "ordinary",
        ),
        AvailabilityRecord(
            "SEC-A",
            20,
            20,
            AvailabilitySnapshot(True, False, False, True, True),
            "delisting",
        ),
    )
    observed = availability_at(
        states,
        security_ids=("SEC-A",),
        effective_at_ms=15,
        knowledge_at_ms=15,
    )["SEC-A"]
    assert observed.decision_eligible
    assert not observed.terminal_event


def test_membership_rejects_future_selected_effective_row() -> None:
    with pytest.raises(PITAlphaDataError, match="unavailable when it became effective"):
        MembershipEvent("SEC-A", 10, 11, 9, True, 1).validate()
    with pytest.raises(PITAlphaDataError, match="future survival"):
        UniverseRule(
            "bad",
            "a" * 64,
            "point-in-time-events",
            63,
            1,
            uses_future_survival=True,
        ).validate()


def test_split_preserves_economic_value() -> None:
    before = EconomicPosition.from_mapping({"SEC-A": 10.0})
    assert mark_position(before, {"SEC-A": 20.0}) == 200.0
    after = apply_corporate_action(
        before,
        CorporateActionRecord(
            "split",
            "SEC-A",
            CorporateActionKind.SPLIT,
            20,
            10,
            share_ratio=2.0,
        ),
    )
    assert mark_position(after, {"SEC-A": 10.0}) == 200.0


def test_cash_dividend_is_booked_exactly_once() -> None:
    position = EconomicPosition.from_mapping({"SEC-A": 5.0})
    dividend = CorporateActionRecord(
        "dividend",
        "SEC-A",
        CorporateActionKind.CASH_DIVIDEND,
        20,
        10,
        cash_per_share=1.25,
    )
    after = apply_corporate_action(position, dividend)
    assert after.cash == 6.25
    with pytest.raises(PITAlphaDataError, match="more than once"):
        apply_corporate_action(after, dividend)


def test_stock_merger_converts_permanent_position() -> None:
    position = EconomicPosition.from_mapping({"SEC-A": 8.0})
    converted = apply_corporate_action(position, _terminal_stock_merger())
    assert converted.as_mapping() == {"SEC-B": 4.0}
    assert converted.cash == 0.0


def test_mixed_stock_merger_preserves_cash_consideration() -> None:
    position = EconomicPosition.from_mapping({"SEC-A": 8.0})
    converted = apply_corporate_action(
        position,
        TerminalEventRecord(
            event_id="mixed-merger",
            security_id="SEC-A",
            kind=TerminalEventKind.MERGER_STOCK,
            effective_at_ms=100,
            available_at_ms=90,
            cash_per_share=1.25,
            successor_security_id="SEC-B",
            successor_ratio=0.5,
        ),
    )
    assert converted.as_mapping() == {"SEC-B": 4.0}
    assert converted.cash == 10.0


def test_delisting_loss_is_not_silently_replaced_with_zero_return() -> None:
    position = EconomicPosition.from_mapping({"SEC-A": 10.0})
    terminal = TerminalEventRecord(
        "delist",
        "SEC-A",
        TerminalEventKind.DELISTING_CASH,
        20,
        20,
        cash_per_share=2.0,
    )
    disposed = apply_corporate_action(position, terminal)
    assert disposed.as_mapping() == {}
    assert disposed.cash == 20.0
    target = compute_post_fill_total_return(
        (
            EconomicValuePoint(0, 10, 10, 100.0, "market"),
            EconomicValuePoint(1, 20, 20, 20.0, "terminal-disposition", True),
            EconomicValuePoint(2, 30, 30, 20.0, "terminal-disposition", True),
        ),
        fill_session_index=0,
        horizon_sessions=2,
    )
    assert target.simple_return == pytest.approx(-0.8)
    assert target.log_return == pytest.approx(-1.6094379124341003)


def test_total_loss_remains_explicit_and_nonfinite_log_is_not_fabricated() -> None:
    target = compute_post_fill_total_return(
        (
            EconomicValuePoint(0, 10, 10, 100.0, "market"),
            EconomicValuePoint(1, 20, 20, 0.0, "terminal-disposition", True),
        ),
        fill_session_index=0,
        horizon_sessions=1,
    )
    assert target.simple_return == -1.0
    assert target.log_return is None
    assert target.terminal_zero_value


def test_missing_bar_requires_validated_fallback_mark() -> None:
    position = EconomicPosition.from_mapping({"SEC-A": 2.0})
    with pytest.raises(PITAlphaDataError, match="no economic mark"):
        mark_position(position, {})
    points = (
        EconomicValuePoint(0, 10, 10, 100.0, "market"),
        EconomicValuePoint(1, 20, 21, 100.0, "validated-fallback"),
        EconomicValuePoint(2, 30, 30, 105.0, "market"),
    )
    target = compute_post_fill_total_return(
        points,
        fill_session_index=0,
        horizon_sessions=2,
    )
    assert target.simple_return == pytest.approx(0.05)


def test_target_starts_after_declared_fill_and_requires_complete_path() -> None:
    points = tuple(
        EconomicValuePoint(index, index * 10, index * 10, value, "market")
        for index, value in enumerate((90.0, 100.0, 110.0, 121.0))
    )
    target = compute_post_fill_total_return(
        points,
        fill_session_index=1,
        horizon_sessions=2,
    )
    assert target.start_value == 100.0
    assert target.end_value == 121.0
    assert target.simple_return == pytest.approx(0.21)
    with pytest.raises(PITAlphaDataError, match="complete economic path"):
        compute_post_fill_total_return(
            (points[1], points[3]),
            fill_session_index=1,
            horizon_sessions=2,
        )


def test_cash_earns_declared_causal_return() -> None:
    position = EconomicPosition.from_mapping({}, cash=1_000.0)
    accrued = apply_cash_return(position, 0.0002)
    assert accrued.cash == pytest.approx(1_000.2)
    assert accrued.as_mapping() == {}


def test_dataset_authority_requires_terminal_disposition_and_permanent_axis() -> None:
    authority = _authority()
    authority.validate()
    invalid = PITAlphaDatasetAuthority(
        manifest=authority.manifest,
        security_master=authority.security_master,
        ticker_history=authority.ticker_history,
        membership_events=authority.membership_events,
        availability=authority.availability,
        corporate_actions=authority.corporate_actions,
        terminal_events=(),
        cash_returns=authority.cash_returns,
    )
    with pytest.raises(PITAlphaDataError, match="every delisted security"):
        invalid.validate()

    missing_cash = PITAlphaDatasetAuthority(
        manifest=authority.manifest,
        security_master=authority.security_master,
        ticker_history=authority.ticker_history,
        membership_events=authority.membership_events,
        availability=authority.availability,
        corporate_actions=authority.corporate_actions,
        terminal_events=authority.terminal_events,
        cash_returns=(),
    )
    with pytest.raises(PITAlphaDataError, match="causal cash returns"):
        missing_cash.validate()


def test_data_gate_requires_independent_complete_event_reconciliation() -> None:
    authority = _authority()
    reconciliations = (
        IndependentEconomicReconciliation(
            event_id="split-a",
            security_id="SEC-A",
            internal_value_change=0.0,
            independent_value_change=0.0,
            absolute_tolerance=1e-12,
            independent_source_receipt_sha256="f" * 64,
        ),
        IndependentEconomicReconciliation(
            event_id="terminal-a",
            security_id="SEC-A",
            internal_value_change=-0.5,
            independent_value_change=-0.5,
            absolute_tolerance=1e-12,
            independent_source_receipt_sha256="f" * 64,
        ),
    )
    passed = evaluate_pit_alpha_data_gate(
        authority,
        reloaded_dataset_receipt_sha256=authority.manifest.receipt_sha256,
        first_tensor_materialization_receipt_sha256="1" * 64,
        second_tensor_materialization_receipt_sha256="1" * 64,
        reconciliations=reconciliations,
    )
    assert passed.passed

    incomplete = evaluate_pit_alpha_data_gate(
        authority,
        reloaded_dataset_receipt_sha256=authority.manifest.receipt_sha256,
        first_tensor_materialization_receipt_sha256="1" * 64,
        second_tensor_materialization_receipt_sha256="1" * 64,
        reconciliations=reconciliations[:1],
    )
    assert not incomplete.passed
    assert incomplete.missing_event_ids == ("terminal-a",)


def test_data_gate_rejects_same_source_as_independent_reconciliation() -> None:
    authority = _authority()
    evidence = evaluate_pit_alpha_data_gate(
        authority,
        reloaded_dataset_receipt_sha256=authority.manifest.receipt_sha256,
        first_tensor_materialization_receipt_sha256="1" * 64,
        second_tensor_materialization_receipt_sha256="1" * 64,
        reconciliations=(
            IndependentEconomicReconciliation(
                "split-a", "SEC-A", 0.0, 0.0, 0.0, "b" * 64
            ),
            IndependentEconomicReconciliation(
                "terminal-a", "SEC-A", -0.5, -0.5, 0.0, "f" * 64
            ),
        ),
    )

    assert not evidence.passed
    assert evidence.independent_source_overlap_event_ids == ("split-a",)


def test_manifest_roundtrip_and_exact_inventory(tmp_path: Path) -> None:
    authority_dir = tmp_path / "authorities"
    authority_dir.mkdir()
    security_path = authority_dir / "security.json"
    security_path.write_bytes(b"{}")
    record = DatasetFileRecord(
        relative_path="authorities/security.json",
        size_bytes=2,
        file_sha256=hashlib.sha256(b"{}").hexdigest(),
        media_type="application/json",
    )
    manifest = _manifest((record,))
    manifest_path = tmp_path / "manifest.json"
    manifest_file_sha = write_pit_alpha_manifest(manifest_path, manifest)
    loaded = load_pit_alpha_manifest(
        manifest_path,
        expected_file_sha256=manifest_file_sha,
    )
    assert loaded == manifest
    validate_manifest_files(tmp_path, loaded)

    security_path.write_bytes(b"[]")
    with pytest.raises(PITAlphaDataError, match="bytes drifted"):
        validate_manifest_files(tmp_path, loaded)


def test_manifest_rejects_unlisted_and_linked_files(tmp_path: Path) -> None:
    data_path = tmp_path / "data.bin"
    data_path.write_bytes(b"alpha")
    manifest = _manifest(
        (
            DatasetFileRecord(
                "data.bin",
                5,
                hashlib.sha256(b"alpha").hexdigest(),
                "application/octet-stream",
            ),
        )
    )
    (tmp_path / "extra.bin").write_bytes(b"unexpected")
    with pytest.raises(PITAlphaDataError, match="unmanifested"):
        validate_manifest_files(tmp_path, manifest, allowed_unlisted=())
    (tmp_path / "extra.bin").unlink()
    (tmp_path / "linked.bin").symlink_to(data_path)
    with pytest.raises(PITAlphaDataError, match="symbolic link"):
        validate_manifest_files(tmp_path, manifest, allowed_unlisted=())


def test_manifest_loader_rejects_self_consistent_wrong_field_types(tmp_path: Path) -> None:
    manifest = _manifest(
        (
            DatasetFileRecord(
                "data.bin",
                5,
                hashlib.sha256(b"alpha").hexdigest(),
                "application/octet-stream",
            ),
        )
    )
    value = manifest.to_dict()
    rule_value = value["universe_rule"]
    assert isinstance(rule_value, dict)
    rule = dict(rule_value)
    rule["uses_future_survival"] = ""
    value["universe_rule"] = rule
    unsigned = {key: row for key, row in value.items() if key != "receipt_sha256"}
    value["receipt_sha256"] = semantic_sha256(unsigned)
    raw = canonical_json_file_bytes(value)
    path = tmp_path / "manifest.json"
    path.write_bytes(raw)
    assert json.loads(raw)["universe_rule"]["uses_future_survival"] == ""
    with pytest.raises(PITAlphaDataError, match="malformed"):
        load_pit_alpha_manifest(
            path,
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )
