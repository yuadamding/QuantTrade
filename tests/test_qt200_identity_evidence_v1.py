from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from rl_quant.data_sources.massive.qt200_identity_evidence_v1 import (
    QT200IdentityEvidenceError,
    build_qt200_identity_evidence_v1,
    build_qt200_issue_resolution_evidence_v1,
    load_qt200_dated_identity_diagnostic_v1,
)

DIGEST = "a" * 64
CAPTURE_MS = 1_787_859_968_859


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _provenance():
    return dict(source_file="provider-capture.json", source_sha256=DIGEST,
                upstream_receipt_sha256="b" * 64, raw_response_body_sha256="c" * 64,
                captured_at_ms=CAPTURE_MS, historical_available_at=None,
                result_index=0, page_index=0)


def _reference(ticker="AAA", *, figi="ISSUE-A", share_class_figi=None):
    provider = dict(ticker=ticker, active=True, cik="0001234567", type="CS",
                    primary_exchange="XNYS", name="Example Corp")
    if figi is not None:
        provider["composite_figi"] = figi
    if share_class_figi is not None:
        provider["share_class_figi"] = share_class_figi
    return dict(provider_record=provider, provenance=_provenance())


def _capture(ticker="AAA", *, figi="ISSUE-A", events=None):
    result = dict(cik="0001234567", events=events or [
        dict(date="2020-01-02", type="ticker_change", ticker_change=dict(ticker=ticker))])
    if figi is not None:
        result["composite_figi"] = figi
    return dict(qt200_tickers=[ticker], request_identifier=figi or ticker,
                provider_response=dict(status="OK", results=result),
                response_metadata=dict(response_status=200, completed_at_ms=CAPTURE_MS),
                provenance=_provenance())


def _dividend(*, ticker="AAA", candidate="AAA", event_id="DIV-A", amount=1.0):
    return dict(candidate_qt200_tickers=[candidate], provenance=_provenance(),
                provider_record=dict(ticker=ticker, id=event_id, cash_amount=amount,
                                     currency="USD", ex_dividend_date="2022-01-03",
                                     pay_date="2022-01-20", record_date="2022-01-04"))


def _bundle(tmp_path: Path, *, references=None, captures=None, dividends=None, splits=None):
    root = tmp_path / "support"
    root.mkdir()
    tickers = ["AAA", "BBB"] + [f"T{index:03d}" for index in range(198)]
    spec = _bytes(dict(ticker_count=200, tickers=tickers))
    payloads = {"qt200-universe-spec.json": spec}
    for name, rows in (
        ("reference-records.jsonl", [_reference()] if references is None else references),
        ("ticker-event-captures.jsonl", [_capture()] if captures is None else captures),
        ("dividends.jsonl", [_dividend()] if dividends is None else dividends),
        ("splits.jsonl", [] if splits is None else splits),
    ):
        payloads[name] = b"".join(_bytes(row) + b"\n" for row in rows)
    inventory = [dict(relative_path=name, bytes=len(data), sha256=_sha(data))
                 for name, data in payloads.items()]
    manifest = dict(files=list(inventory), universe_sha256=_sha(spec), sources=[dict(
        source_file="provider-capture.json", source_sha256=DIGEST,
        upstream_receipt_sha256="b" * 64)])
    payloads["manifest.json"] = _bytes(manifest)
    inventory.append(dict(relative_path="manifest.json", bytes=len(payloads["manifest.json"]),
                          sha256=_sha(payloads["manifest.json"])))
    inventory.sort(key=lambda row: row["relative_path"])
    complete = _bytes(dict(
        schema="quanttrade-qt200-support-extraction-complete-v1",
        status="COMPLETE_SUPPORT_DATA_EXTRACTION_ONLY", inventory=inventory,
        inventory_sha256=_sha(_bytes(inventory)), manifest_sha256=_sha(payloads["manifest.json"]),
        inventoried_file_count_excluding_complete=len(inventory),
        inventoried_bytes_excluding_complete=sum(row["bytes"] for row in inventory)))
    payloads["COMPLETE.json"] = complete
    for name, data in payloads.items():
        (root / name).write_bytes(data)
    return dict(support_root=root, expected_completion_sha256=_sha(complete),
                expected_universe_sha256=_sha(spec))


def test_source_bound_candidates_never_mint_historical_or_training_authority(tmp_path):
    result = build_qt200_identity_evidence_v1(**_bundle(tmp_path))
    assert len(result["ordered_tickers"]) == 200
    assert result["security_candidates"][0]["current_issue_identifier_observed"]
    assert result["security_candidates"][0]["native_security_id"] is None
    assert result["native_security_ids_emitted"] == 0
    for key in ("historical_identity_qualified", "economic_accounting_qualified",
                "strict_pit_qualified", "finalized_accounting_research_authorized",
                "training_ready_for_adaptive_v5", "original_provider_raw_bodies_reauthenticated"):
        assert result[key] is False
    assert result["economic_observations"][0]["historical_available_at_ms"] is None
    assert result["economic_observations"][0]["source"]["captured_at_ms"] == CAPTURE_MS
    unsigned = {key: value for key, value in result.items() if key != "receipt_sha256"}
    assert result["receipt_sha256"] == _sha(_bytes(unsigned))


def test_cik_and_one_ticker_event_do_not_resolve_missing_issue_identity(tmp_path):
    result = build_qt200_identity_evidence_v1(**_bundle(
        tmp_path, references=[_reference(figi=None)], captures=[_capture(figi=None)]))
    row = result["security_candidates"][0]
    assert row["current_issuer_cik"] == "0001234567"
    assert not row["current_issue_identifier_observed"]
    assert "issue_level_identifier_missing_cik_is_not_substitute" in row["gaps"]
    assert not row["ticker_event_observations"][0]["same_current_composite_figi"]
    assert "AAA" in result["issue_identifier_missing_tickers"]


def test_figi_is_not_universal_requirement_but_alternate_identifier_is_not_history(tmp_path):
    result = build_qt200_identity_evidence_v1(**_bundle(
        tmp_path, references=[_reference(figi=None, share_class_figi="SHARE-A")],
        captures=[_capture(figi=None)]))
    row = result["security_candidates"][0]
    assert row["current_issue_identifier_observed"]
    assert "AAA" not in result["issue_identifier_missing_tickers"]
    assert row["native_security_id"] is None
    assert not row["historical_identity_qualified"]


def test_old_ticker_reuse_is_not_joined_beyond_explicit_event_bracket(tmp_path):
    events = [dict(date="2020-01-02", type="ticker_change", ticker_change=dict(ticker="OLD")),
              dict(date="2021-01-04", type="ticker_change", ticker_change=dict(ticker="AAA"))]
    result = build_qt200_identity_evidence_v1(**_bundle(
        tmp_path, captures=[_capture(events=events)], dividends=[_dividend(ticker="OLD")]))
    join = result["economic_observations"][0]["candidate_joins"][0]
    assert join["candidate_state"] == "alias_outside_provider_event_bracket_or_unlinked"
    assert not join["event_bracket_compatible"]
    assert not join["identity_qualified"]


def test_alias_inside_bracket_is_still_only_retrospective_candidate(tmp_path):
    events = [dict(date="2020-01-02", type="ticker_change", ticker_change=dict(ticker="OLD")),
              dict(date="2023-01-03", type="ticker_change", ticker_change=dict(ticker="AAA"))]
    result = build_qt200_identity_evidence_v1(**_bundle(
        tmp_path, captures=[_capture(events=events)], dividends=[_dividend(ticker="OLD")]))
    join = result["economic_observations"][0]["candidate_joins"][0]
    assert join["event_bracket_compatible"]
    assert join["candidate_state"] == "same_figi_event_bracket_candidate_only"
    assert not join["identity_qualified"]


def test_current_ticker_conflict_and_duplicate_event_dates_are_recorded(tmp_path):
    events = [dict(date="2020-01-02", type="ticker_change", ticker_change=dict(ticker="AAA")),
              dict(date="2020-01-02", type="ticker_change", ticker_change=dict(ticker="OTHER"))]
    result = build_qt200_identity_evidence_v1(**_bundle(tmp_path, captures=[_capture(events=events)]))
    row = result["security_candidates"][0]
    assert "duplicate_or_conflicting_provider_event_dates" in row["gaps"]
    assert "current_ticker_differs_from_last_provider_event" in row["gaps"]
    assert len(row["ticker_event_observations"]) == 2


def test_conflicting_economic_ids_are_preserved_without_silent_deduplication(tmp_path):
    result = build_qt200_identity_evidence_v1(**_bundle(
        tmp_path, dividends=[_dividend(amount=1.0), _dividend(amount=2.0)]))
    assert len(result["economic_observations"]) == 2
    conflict = result["duplicate_economic_event_observations"][0]
    assert conflict["kind"] == "conflicting_provider_event_id"
    assert conflict["first_record_index"] == 0
    assert conflict["record_index"] == 1


def test_zero_records_are_not_a_negative_corporate_action_authority(tmp_path):
    result = build_qt200_identity_evidence_v1(**_bundle(tmp_path, dividends=[], splits=[]))
    assert result["economic_observations"] == []
    assert not result["economic_accounting_qualified"]
    assert not result["security_candidates"][0]["absence_of_corporate_action_records_proves_no_events"]


def test_payload_tamper_is_rejected(tmp_path):
    args = _bundle(tmp_path)
    (args["support_root"] / "reference-records.jsonl").write_bytes(b"{}\n")
    with pytest.raises(QT200IdentityEvidenceError, match="hash/size mismatch"):
        build_qt200_identity_evidence_v1(**args)


def test_symlink_even_to_matching_payload_is_rejected(tmp_path):
    args = _bundle(tmp_path)
    path = args["support_root"] / "reference-records.jsonl"
    original = tmp_path / "original.jsonl"
    path.rename(original)
    path.symlink_to(original)
    with pytest.raises(QT200IdentityEvidenceError, match="Symlink"):
        build_qt200_identity_evidence_v1(**args)


def test_record_source_cross_link_is_checked_even_with_valid_bundle_hashes(tmp_path):
    row = _reference()
    row["provenance"]["source_sha256"] = "d" * 64
    args = _bundle(tmp_path, references=[row])
    with pytest.raises(QT200IdentityEvidenceError, match="provenance"):
        build_qt200_identity_evidence_v1(**args)


def test_duplicate_json_keys_are_rejected(tmp_path):
    args = _bundle(tmp_path)
    payload = b'{"schema":"a","schema":"b"}'
    (args["support_root"] / "COMPLETE.json").write_bytes(payload)
    args["expected_completion_sha256"] = _sha(payload)
    with pytest.raises(QT200IdentityEvidenceError, match="Duplicate JSON key"):
        build_qt200_identity_evidence_v1(**args)


def _dated_response(*, day="2022-01-03", status=200, figi="ISSUE-A", cik="111", list_date="2020-01-02"):
    result = dict(ticker="AAA", composite_figi=figi, cik=cik, list_date=list_date,
                  name="Example", primary_exchange="XNYS", type="CS", active=True) if status == 200 else None
    payload = dict(status="OK" if status == 200 else "NOT_FOUND", request_id="REQUEST-A")
    if result is not None:
        payload["results"] = result
    body = _bytes(payload)
    return dict(ticker="AAA", date=day,
                request_url="https://api.massive.com/v3/reference/tickers/AAA?date=" + day,
                http_status=status, requested_at_ms=CAPTURE_MS - 1, completed_at_ms=CAPTURE_MS,
                provider_request_id="REQUEST-A", raw_response_body_base64=base64.b64encode(body).decode(),
                raw_response_body_sha256=_sha(body), raw_response_content_length=len(body),
                results=result, historical_known_at=None, capture_time_is_historical_availability=False,
                absent_response_proves_delisting=False)


def _dated_bundle(tmp_path, rows=None, *, universe_sha256=DIGEST):
    root = tmp_path / "dated"
    root.mkdir()
    rows = [_dated_response()] if rows is None else rows
    source = b"# Frozen acquisition source is hashed, never executed by the loader.\n"
    (root / "capture.py").write_bytes(source)
    plan = _bytes(dict(schema="qt200-dated-identity-capture-plan-v1",
                       requests=[[row["ticker"], row["date"]] for row in rows],
                       requested_count=len(rows), source_sha256=_sha(source),
                       universe_sha256=universe_sha256, production_qualification=False))
    (root / "plan.json").write_bytes(plan)
    entries, statuses = [], {}
    for index, row in enumerate(rows):
        body = _bytes(row)
        path = f"response-{index:04d}.json"
        (root / path).write_bytes(body)
        entries.append(dict(index=index, ticker=row["ticker"], date=row["date"],
                            path=path, bytes=len(body), sha256=_sha(body)))
        code = str(row["http_status"])
        statuses[code] = statuses.get(code, 0) + 1
    completion = _bytes(dict(schema="qt200-dated-identity-capture-completion-v1",
                            plan=dict(path="plan.json", bytes=len(plan), sha256=_sha(plan)),
                            requested_count=len(rows), completed_count=len(rows), files=entries,
                            failure_indices=[], http_status_counts=statuses,
                            identity_qualified=False, point_in_time_qualified=False, training_ready=False))
    (root / "COMPLETE.json").write_bytes(completion)
    return dict(capture_root=root, expected_completion_sha256=_sha(completion),
                expected_universe_sha256=universe_sha256)


def test_dated_200_and_404_are_observations_not_intervals_or_delisting(tmp_path):
    rows = [_dated_response(day="2017-01-03", status=404), _dated_response()]
    result = load_qt200_dated_identity_diagnostic_v1(**_dated_bundle(tmp_path, rows))
    assert result["observed_http_status_counts"] == {"200": 1, "404": 1}
    assert result["observations"][0]["provider_fields"] is None
    assert not result["absence_proves_delisting"]
    assert not result["sparse_snapshots_prove_continuous_listing"]
    assert not result["historical_identity_qualified"]
    assert not result["point_in_time_qualified"]
    assert not result["training_ready_for_adaptive_v5"]


def test_dated_issue_and_issuer_changes_do_not_infer_successor(tmp_path):
    rows = [_dated_response(), _dated_response(day="2026-08-26", figi="ISSUE-B", cik="222")]
    result = load_qt200_dated_identity_diagnostic_v1(**_dated_bundle(tmp_path, rows))
    change = result["observed_field_changes"][0]
    assert change["fields"]["composite_figi"] == {"before": "ISSUE-A", "after": "ISSUE-B"}
    assert change["fields"]["cik"] == {"before": "111", "after": "222"}
    assert not change["successor_or_corporate_action_inferred"]
    assert result["native_security_ids_emitted"] == 0


def test_dated_list_date_and_capture_chronology_conflicts_are_reported(tmp_path):
    row = _dated_response(list_date="2023-01-03")
    row["requested_at_ms"] = CAPTURE_MS + 1
    result = load_qt200_dated_identity_diagnostic_v1(**_dated_bundle(tmp_path, [row]))
    assert "provider_list_date_after_query_date" in result["observations"][0]["issues"]
    assert "capture_completed_before_request" in result["observations"][0]["issues"]


def test_dated_raw_body_hash_is_checked_independently_of_response_file_hash(tmp_path):
    row = _dated_response()
    row["raw_response_body_sha256"] = "d" * 64
    with pytest.raises(QT200IdentityEvidenceError, match="raw-body physical hash"):
        load_qt200_dated_identity_diagnostic_v1(**_dated_bundle(tmp_path, [row]))


def test_dated_decoded_results_must_match_exact_raw_body(tmp_path):
    row = _dated_response()
    row["results"]["cik"] = "FORGED"
    with pytest.raises(QT200IdentityEvidenceError, match="parsed response/raw-body"):
        load_qt200_dated_identity_diagnostic_v1(**_dated_bundle(tmp_path, [row]))


def test_dated_duplicate_request_pairs_cannot_pass_coverage(tmp_path):
    with pytest.raises(QT200IdentityEvidenceError, match="Duplicate dated request pair"):
        load_qt200_dated_identity_diagnostic_v1(**_dated_bundle(tmp_path, [_dated_response(), _dated_response()]))


def test_dated_capture_source_hash_is_bound_not_executed(tmp_path):
    args = _dated_bundle(tmp_path)
    (args["capture_root"] / "capture.py").write_bytes(b"raise RuntimeError('not executed')\n")
    with pytest.raises(QT200IdentityEvidenceError, match="physical hash/size mismatch"):
        load_qt200_dated_identity_diagnostic_v1(**args)


def _issue_response(*, ticker="AAA", day="2022-01-03", figi="ISSUE-A",
                    share_class_figi=None, status=200, name="Example Corp", cik="0001234567"):
    row = _dated_response(day=day, figi=figi, status=status, cik=cik)
    row["ticker"] = ticker
    row["request_url"] = "https://api.massive.com/v3/reference/tickers/" + ticker + "?date=" + day
    payload = dict(status="OK" if status == 200 else "NOT_FOUND", request_id="REQUEST-A")
    if status == 200:
        row["results"].update(ticker=ticker, name=name, share_class_figi=share_class_figi)
        payload["results"] = row["results"]
    body = _bytes(payload)
    row.update(raw_response_body_base64=base64.b64encode(body).decode(),
               raw_response_body_sha256=_sha(body), raw_response_content_length=len(body))
    return row


def _issue_bundle(tmp_path, *, rows=None, **support_kwargs):
    support = _bundle(tmp_path, **support_kwargs)
    dated = _dated_bundle(tmp_path, rows, universe_sha256=support["expected_universe_sha256"])
    return dict(support_root=support["support_root"],
                expected_support_completion_sha256=support["expected_completion_sha256"],
                capture_root=dated["capture_root"],
                expected_capture_completion_sha256=dated["expected_completion_sha256"],
                expected_universe_sha256=support["expected_universe_sha256"])


def test_issue_review_preserves_panel_order_and_never_promotes_observed_identifier_support(tmp_path):
    result = build_qt200_issue_resolution_evidence_v1(**_issue_bundle(tmp_path))
    assert [row["qt200_ticker"] for row in result["security_resolutions"]] == result["ordered_tickers"]
    assert len(result["security_resolutions"]) == 200
    assertion = result["security_resolutions"][0]["dated_assertions"][0]
    assert assertion["classification"] == "supports_current_observed_issue_identifiers"
    assert assertion["matching_identifier_fields"] == ["composite_figi"]
    assert assertion["captured_at_ms"] == CAPTURE_MS
    assert assertion["query_date"] == "2022-01-03"
    assert assertion["historical_known_at_ms"] is None
    assert not assertion["identity_qualified"]
    assert not assertion["continuous_interval_inferred"]
    assert result["native_security_ids_emitted"] == result["successor_joins_emitted"] == 0
    for flag in ("historical_identity_qualified", "point_in_time_qualified", "economic_accounting_qualified",
                 "training_ready_for_adaptive_v5", "support_original_provider_raw_bodies_reauthenticated"):
        assert result[flag] is False
    unsigned = {key: value for key, value in result.items() if key != "receipt_sha256"}
    assert result["receipt_sha256"] == _sha(_bytes(unsigned))


@pytest.mark.parametrize("figi,share,conflict,match", [
    ("ISSUE-B", "SHARE-A", "composite_figi", "share_class_figi"),
    ("ISSUE-A", "SHARE-B", "share_class_figi", "composite_figi"),
])
def test_issue_review_conflicting_identifier_overrides_other_identifier_match(tmp_path, figi, share, conflict, match):
    result = build_qt200_issue_resolution_evidence_v1(**_issue_bundle(
        tmp_path, references=[_reference(share_class_figi="SHARE-A")],
        rows=[_issue_response(figi=figi, share_class_figi=share)]))
    assertion = result["security_resolutions"][0]["dated_assertions"][0]
    assert assertion["classification"] == "conflicts_with_current_observed_issue_identifiers"
    assert assertion["conflicting_identifier_fields"] == [conflict]
    assert assertion["matching_identifier_fields"] == [match]
    assert result["dated_observations"][0]["provider_fields"][conflict] in ("ISSUE-B", "SHARE-B")


@pytest.mark.parametrize("current,observed", [(None, "ISSUE-A"), ("ISSUE-A", None)])
def test_issue_review_matching_name_and_cik_cannot_resolve_missing_issue_identifiers(tmp_path, current, observed):
    result = build_qt200_issue_resolution_evidence_v1(**_issue_bundle(
        tmp_path, references=[_reference(figi=current)], rows=[_issue_response(figi=observed)]))
    assertion = result["security_resolutions"][0]["dated_assertions"][0]
    assert assertion["classification"] == "unknown_issue_identity"
    assert "no_comparable_issue_identifier_cik_and_name_are_insufficient" in assertion["reasons"]
    assert not result["name_or_cik_used_as_issue_identifier"]


def test_issue_review_ambiguous_current_reference_remains_unknown_even_when_one_row_matches(tmp_path):
    result = build_qt200_issue_resolution_evidence_v1(**_issue_bundle(
        tmp_path, references=[_reference(), _reference(figi="ISSUE-B")]))
    resolution = result["security_resolutions"][0]
    assert resolution["dated_assertions"][0]["classification"] == "unknown_issue_identity"
    assert "unique_current_reference_not_established" in resolution["review_flags"]
    assert len(resolution["current_reference_evidence"]["current_reference_sources"]) == 2


def test_issue_review_404_and_unqueried_aliases_are_unknown_not_negative_identity_evidence(tmp_path):
    events = [dict(date="2020-01-02", type="ticker_change", ticker_change=dict(ticker="OLD")),
              dict(date="2021-01-04", type="ticker_change", ticker_change=dict(ticker="AAA"))]
    result = build_qt200_issue_resolution_evidence_v1(**_issue_bundle(
        tmp_path, captures=[_capture(events=events)], rows=[_issue_response(status=404)]))
    resolution = result["security_resolutions"][0]
    assert resolution["query_symbols_without_observations"] == ["OLD"]
    assertion = resolution["dated_assertions"][0]
    assert assertion["classification"] == "unknown_issue_identity"
    assert "unsuccessful_query_is_not_issue_or_delisting_evidence" in assertion["reasons"]
    assert not result["dated_observations"][0]["delisting_proven"]


def test_issue_review_alias_contradiction_is_attached_to_economic_candidate_not_used_as_mask(tmp_path):
    events = [dict(date="2020-01-02", type="ticker_change", ticker_change=dict(ticker="OLD"))]
    result = build_qt200_issue_resolution_evidence_v1(**_issue_bundle(
        tmp_path, captures=[_capture(events=events)], dividends=[_dividend(ticker="OLD")],
        rows=[_issue_response(ticker="OLD", figi="OTHER-CLASS", share_class_figi="OTHER-SHARE")]))
    resolution = result["security_resolutions"][0]
    assert "current_ticker_differs_from_last_provider_event" in resolution["review_flags"]
    assert "alias_query_conflicts_with_current_issue_identifiers" in resolution["review_flags"]
    join = result["economic_observations"][0]["candidate_joins"][0]
    assert join["candidate_state"] == "same_figi_event_bracket_candidate_only"
    assert join["exact_event_date_dated_assertions"][0]["conflicting_identifier_fields"] == ["composite_figi"]
    assert join["review_flags_are_not_event_date_validity_masks"]
    assert join["dated_assertions_do_not_authorize_economic_join"]
    assert not join["identity_qualified"]


def test_issue_review_off_date_conflict_does_not_become_an_effective_date_join(tmp_path):
    result = build_qt200_issue_resolution_evidence_v1(**_issue_bundle(
        tmp_path, rows=[_issue_response(day="2026-08-26", figi="OTHER")]))
    join = result["economic_observations"][0]["candidate_joins"][0]
    assert "conflicting_dated_issue_identifiers_require_source_resolution" in join["cross_capture_review_flags"]
    assert join["exact_event_date_dated_assertions"] == []
    assert not result["economic_observations"][0]["accounting_authorized"]


def test_issue_review_unassigned_queries_and_original_claims_remain_visible(tmp_path):
    result = build_qt200_issue_resolution_evidence_v1(**_issue_bundle(
        tmp_path, rows=[_issue_response(ticker="UNRELATED", figi="OTHER", name="Different issuer")]))
    assert result["unassigned_dated_observation_indices"] == [0]
    assert result["dated_observations"][0]["provider_fields"]["name"] == "Different issuer"
    assert result["security_resolutions"][0]["dated_assertions"] == []


@pytest.mark.parametrize("value", [123, " ISSUE-A "])
def test_issue_review_malformed_identifier_is_retained_but_not_compared(tmp_path, value):
    result = build_qt200_issue_resolution_evidence_v1(**_issue_bundle(
        tmp_path, rows=[_issue_response(figi=value)]))
    assertion = result["security_resolutions"][0]["dated_assertions"][0]
    assert assertion["observed_issue_identifiers"]["composite_figi"] == value
    assert assertion["classification"] == "unknown_issue_identity"
    assert "malformed_or_empty_issue_identifier_not_compared" in assertion["reasons"]


def test_issue_review_reopens_support_hash_instead_of_accepting_caller_report(tmp_path):
    args = _issue_bundle(tmp_path)
    (args["support_root"] / "reference-records.jsonl").write_bytes(b"{}\n")
    with pytest.raises(QT200IdentityEvidenceError, match="physical hash/size mismatch"):
        build_qt200_issue_resolution_evidence_v1(**args)


def test_issue_review_reopens_dated_raw_body_hash_and_shared_universe(tmp_path):
    row = _issue_response()
    row["raw_response_body_sha256"] = "d" * 64
    args = _issue_bundle(tmp_path, rows=[row])
    with pytest.raises(QT200IdentityEvidenceError, match="raw-body physical hash"):
        build_qt200_issue_resolution_evidence_v1(**args)
    args["expected_universe_sha256"] = "e" * 64
    with pytest.raises(QT200IdentityEvidenceError, match="Universe physical hash"):
        build_qt200_issue_resolution_evidence_v1(**args)
