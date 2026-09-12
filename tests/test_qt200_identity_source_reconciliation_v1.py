from __future__ import annotations

import base64
import copy
import json
import os
from datetime import datetime, timezone
from urllib.parse import urlencode

import pytest

from rl_quant.data_sources.massive.qt200_identity_evidence_v1 import (
    MAX_FILE_BYTES,
    QT200IdentityEvidenceError,
    reconcile_qt200_identity_source_v1,
)
from test_qt200_split_source_reconciliation_v1 import _bytes, _inventory, _publish_support, _sha

NAMES = ("reference-tickers-current-v1.json", "ticker-events-v1.json", "missing-figi-ticker-events-v1.json")
SCHEMAS = ("quanttrade-massive-reference-tickers-capture-v1", "quanttrade-massive-ticker-events-capture-v1",
           "quanttrade-massive-missing-figi-ticker-events-capture-v1")
QUERIES = ("us-stocks-current-active-and-inactive-complete-pagination",
           "ticker-change-events-by-distinct-common-stock-composite-figi",
           "ticker-change-events-by-missing-figi-common-stock-ticker")
STAMP = 1_800_000_000_000


def _canonical(value):
    return _bytes(value) + b"\n"


def _seal(document):
    document = {k: v for k, v in document.items() if k != "receipt_sha256"}
    return dict(document, receipt_sha256=_sha(_canonical(document)))


def _response(body, ordinal):
    body = dict(body, request_id=f"REQUEST-{ordinal}")
    raw = _canonical(body)
    return dict(requested_at_ms=STAMP+ordinal*10, completed_at_ms=STAMP+ordinal*10+1,
        provider_request_id=body["request_id"], response_content_type="application/json",
        raw_response_body_base64=base64.b64encode(raw).decode(),
        raw_response_body_sha256=_sha(raw), raw_response_content_length=len(raw))


def _provenance(source, response, index, capture_index=None):
    return dict(source_file=source["source_file"], source_sha256=source["source_sha256"],
        upstream_receipt_sha256=source["upstream_receipt_sha256"], capture_index=capture_index,
        page_index=response.get("page_index"), result_index=index,
        provider_request_id=response["provider_request_id"], raw_response_body_sha256=response["raw_response_body_sha256"],
        captured_at_ms=response["completed_at_ms"],
        captured_at_utc=datetime.fromtimestamp(response["completed_at_ms"]/1000, timezone.utc).isoformat(),
        historical_available_at=None)


def _fixture(tmp_path, *, duplicate_reference=False, duplicate_event=False, multi_target=False):
    source = tmp_path / "original"
    support = tmp_path / "support"
    source.mkdir()
    support.mkdir()
    tickers = ["AAA", "BBB"] + [f"T{i:03d}" for i in range(198)]
    old = dict(ticker="OLD", active=False, type="CS", composite_figi="FIGI-A", name="Café Old")
    inactive = [old, dict(ticker="OUT", active=False, type="CS" if multi_target else "ETF")]
    if duplicate_reference:
        inactive.insert(1, copy.deepcopy(old))
    active = [dict(ticker="AAA", active=True, type="CS", composite_figi="FIGI-A"),
              dict(ticker="BBB", active=True, type="CS")]
    if multi_target:
        active[1]["composite_figi"] = "FIGI-A"
    captures = []
    for index, rows in enumerate((inactive, active)):
        url = "https://api.massive.com/v3/reference/tickers?" + urlencode(dict(
            market="stocks", locale="us", active=str(bool(index)).lower(), limit="1000", sort="ticker", order="asc"))
        page = dict(_response(dict(status="OK", results=rows), index), page_index=0,
            request_url=url, final_response_url=url, result_count=len(rows),
            result_inventory_sha256=_sha(_canonical(rows)), next_url=None)
        captures.append(dict(active=bool(index), page_count=1, result_count=len(rows), pages=[page]))
    producers = {name: str(i+1)*64 for i, name in enumerate(NAMES)}
    reference = _seal(dict(schema=SCHEMAS[0], capture_id="synthetic-reference", query=QUERIES[0],
        endpoint="https://api.massive.com/v3/reference/tickers", captures=captures,
        entitlement_receipt_sha256="a"*64, implementation_source_sha256=producers[NAMES[0]],
        credential_values_recorded=False))
    reference_sha = _sha(_canonical(reference))
    events = [dict(date="2022-01-01", type="ticker_change", ticker_change=dict(ticker="OLD")),
              dict(date="2022-06-01", type="ticker_change", ticker_change=dict(ticker="AAA"))]
    if duplicate_event:
        events.insert(1, copy.deepcopy(events[0]))
    response = dict(_response(dict(status="OK", results=dict(composite_figi="FIGI-A", events=events)), 2),
                    identifier="FIGI-A", response_status=200, event_count=len(events))
    missing = dict(_response(dict(status="NOT_FOUND", message="No events found for given ID"), 3),
                   identifier="OUT" if multi_target else "BBB", response_status=404, event_count=0)
    documents = {NAMES[0]: reference}
    for index, row in enumerate((response, missing), 1):
        document = dict(schema=SCHEMAS[index], capture_id=f"synthetic-events-{index}", query=QUERIES[index],
            endpoint="https://api.massive.com/vX/reference/tickers", rows=[row],
            reference_capture_file_sha256=reference_sha,
            identifier_inventory_sha256=_sha(_canonical([row["identifier"]])),
            entitlement_receipt_sha256="a"*64, implementation_source_sha256=producers[NAMES[index]],
            credential_values_recorded=False)
        if index == 2:
            document["request_implementation_source_sha256"] = producers[NAMES[1]]
        documents[NAMES[index]] = _seal(document)
    sources = []
    for name, document in documents.items():
        raw = _canonical(document)
        (source / name).write_bytes(raw)
        sources.append(dict(source_file=name, source_bytes=len(raw), source_sha256=_sha(raw),
                            upstream_receipt_sha256=document["receipt_sha256"]))
    source_map = {row["source_file"]: row for row in sources}
    selected_references = []
    for capture_index, rows in enumerate((inactive, active)):
        for index, row in enumerate(rows):
            if row["ticker"] == "OUT":
                continue
            matched = ["AAA", "BBB"] if multi_target else (["BBB"] if row["ticker"] == "BBB" else ["AAA"])
            selected_references.append(dict(qt200_tickers=matched, provider_record=copy.deepcopy(row),
                provenance=_provenance(source_map[NAMES[0]], captures[capture_index]["pages"][0], index, capture_index),
                current_snapshot_only=True, same_composite_figi_qt200_tickers=[] if matched == ["BBB"] else matched,
                historical_identity_from_symbol_match_verified=False))
    selected_captures = []
    for name, row, ticker in ((NAMES[1], response, "AAA"), (NAMES[2], missing, "BBB")):
        if multi_target and name == NAMES[2]:
            continue
        provenance = _provenance(source_map[name], row, 0)
        metadata = {k: v for k, v in row.items() if k != "raw_response_body_base64"}
        metadata.update(captured_at_utc=provenance["captured_at_utc"], capture_time_is_historical_availability_time=False)
        selected_captures.append(dict(qt200_tickers=["AAA", "BBB"] if multi_target else [ticker], request_identifier=row["identifier"],
            response_metadata=metadata, provider_response=json.loads(base64.b64decode(row["raw_response_body_base64"])),
            provenance=provenance, point_in_time_complete=False))
    provenance = selected_captures[0]["provenance"]
    event_rows = [dict(qt200_ticker=ticker, provider_event=copy.deepcopy(event), composite_figi="FIGI-A",
        permanent_identity_match_verified=not multi_target, event_index=i, provenance=provenance)
        for ticker in (["AAA", "BBB"] if multi_target else ["AAA"]) for i, event in enumerate(events)]
    aliases = []
    for alias, event_date, next_day in (("OLD", "2022-01-01", "2022-06-01"), ("AAA", "2022-06-01", None)):
        if multi_target:
            continue
        aliases.append(dict(qt200_ticker="AAA", provider_ticker=alias, composite_figi="FIGI-A",
            provider_event_date=event_date, next_provider_event_date=next_day,
            candidate_interval_start_inclusive=event_date, candidate_interval_end_exclusive=next_day or "2023-01-01",
            candidate_interval_overlaps_requested_dates=True, current_ticker=alias == "AAA",
            evidence="explicit_provider_ticker_change_same_composite_figi", provenance=provenance,
            permanent_identity_match_verified=True, interval_continuity_verified=False,
            historical_tradability_verified=False, automatic_price_history_stitching_authorized=False,
            conflicting_event_dates=False))
    alias_payload = dict(records=aliases,
        rule="Only explicit provider ticker_change events joined by exact current composite FIGI.",
        interval_semantics="Candidate date bounds, not verified continuous listing/tradability.",
        merger_spinoff_share_class_continuity_inferred=False,
        automatic_price_history_stitching_authorized=False, training_ready=False, PIT=False)
    files = {"qt200-universe-spec.json": _bytes(dict(ticker_count=200, tickers=tickers)),
             "historical-aliases.json": _bytes(alias_payload), "dividends.jsonl": b"", "splits.jsonl": b""}
    for name, rows in (("reference-records.jsonl", selected_references), ("ticker-event-captures.jsonl", selected_captures),
                       ("ticker-change-events.jsonl", event_rows)):
        files[name] = b"".join(_bytes(row)+b"\n" for row in rows)
    args = dict(source_root=source, expected_capture_sha256={r["source_file"]: r["source_sha256"] for r in sources},
        expected_implementation_sha256=producers, expected_entitlement_receipt_sha256="a"*64,
        support_root=support, expected_support_completion_sha256=_publish_support(support, files, sources),
        expected_universe_sha256=_sha(files["qt200-universe-spec.json"]))
    return args, files, sources, documents


def _resign_support(fixture, name, value):
    args, files, sources, _ = fixture
    files[name] = b"".join(_bytes(row)+b"\n" for row in value) if name.endswith(".jsonl") else _bytes(value)
    args["expected_support_completion_sha256"] = _publish_support(args["support_root"], files, sources)
    return args


def _resign_source(fixture, name, *, seal=True):
    args, files, sources, docs = fixture
    if seal:
        docs[name] = _seal(docs[name])
    raw = _canonical(docs[name])
    (args["source_root"] / name).write_bytes(raw)
    args["expected_capture_sha256"][name] = _sha(raw)
    for row in sources:
        if row["source_file"] == name:
            row.update(source_bytes=len(raw), source_sha256=_sha(raw), upstream_receipt_sha256=docs[name]["receipt_sha256"])
    if name == NAMES[0]:
        # Keep dependent source links authentic when probing a deeper original
        # reference invariant; the support observations themselves stay frozen.
        for dependent in NAMES[1:]:
            docs[dependent]["reference_capture_file_sha256"] = _sha(raw)
            _resign_source(fixture, dependent)
    args["expected_support_completion_sha256"] = _publish_support(args["support_root"], files, sources)
    return args


def test_original_identity_reconciliation_is_read_only_and_nonauthorizing(tmp_path):
    args, _, _, _ = _fixture(tmp_path)
    before = _inventory(tmp_path)
    result = reconcile_qt200_identity_source_v1(**args)
    assert _inventory(tmp_path) == before
    assert result["original_reference_result_count"] == 4
    assert result["original_reference_page_count"] == 2
    assert result["selected_reference_count"] == 3
    assert result["selected_event_capture_count"] == result["selected_ticker_change_count"] == result["selected_alias_count"] == 2
    assert result["identity_original_provider_raw_bodies_reauthenticated"] is True
    assert result["original_capture_self_receipts_reauthenticated"] is True
    for name in ("native_source_transactions_present", "third_party_signature_verified", "producer_execution_attested",
                 "all_support_original_provider_raw_bodies_reauthenticated", "historical_identity_qualified",
                 "historical_availability_qualified", "economic_accounting_qualified", "point_in_time_qualified",
                 "training_ready_for_adaptive_v5", "source_writes"):
        assert result[name] is False
    assert result["provider_calls"] == result["native_security_ids_emitted"] == result["successor_joins_emitted"] == 0
    assert result["receipt_sha256"] == _sha(_bytes({k: v for k, v in result.items() if k != "receipt_sha256"}))
    assert reconcile_qt200_identity_source_v1(**args) == result


@pytest.mark.parametrize("kind", ["reference", "event"])
def test_distinct_original_positions_with_identical_values_are_preserved(tmp_path, kind):
    args, _, _, _ = _fixture(tmp_path, duplicate_reference=kind == "reference", duplicate_event=kind == "event")
    result = reconcile_qt200_identity_source_v1(**args)
    assert result["selected_reference_count"] == (4 if kind == "reference" else 3)
    assert result["selected_ticker_change_count"] == (3 if kind == "event" else 2)
    assert result["selected_alias_count"] == 2


@pytest.mark.parametrize("name", NAMES)
def test_changed_original_identity_bytes_fail_exact_hash(tmp_path, name):
    args, _, _, _ = _fixture(tmp_path)
    path = args["source_root"] / name
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(QT200IdentityEvidenceError, match="physical hash"):
        reconcile_qt200_identity_source_v1(**args)


@pytest.mark.parametrize("field,value", [("schema", "wrong"), ("credential_values_recorded", True),
    ("implementation_source_sha256", "f"*64), ("entitlement_receipt_sha256", "f"*64),
    ("endpoint", "https://example.org/vX/reference/tickers")])
def test_resealed_original_identity_cannot_change_capture_contract(tmp_path, field, value):
    fixture = _fixture(tmp_path)
    fixture[3][NAMES[1]][field] = value
    with pytest.raises(QT200IdentityEvidenceError):
        reconcile_qt200_identity_source_v1(**_resign_source(fixture, NAMES[1]))


def test_original_identity_self_receipt_requires_producer_newline(tmp_path):
    fixture = _fixture(tmp_path)
    document = fixture[3][NAMES[1]]
    document["receipt_sha256"] = _sha(_bytes({k: v for k, v in document.items() if k != "receipt_sha256"}))
    with pytest.raises(QT200IdentityEvidenceError, match="self-receipt"):
        reconcile_qt200_identity_source_v1(**_resign_source(fixture, NAMES[1], seal=False))


def test_original_identity_event_reference_cross_link_is_required(tmp_path):
    fixture = _fixture(tmp_path)
    fixture[3][NAMES[1]]["reference_capture_file_sha256"] = "f"*64
    with pytest.raises(QT200IdentityEvidenceError, match="event/reference"):
        reconcile_qt200_identity_source_v1(**_resign_source(fixture, NAMES[1]))


@pytest.mark.parametrize("field,value", [("raw_response_body_sha256", "f"*64),
    ("raw_response_content_length", 1), ("provider_request_id", "different"),
    ("completed_at_ms", STAMP-1), ("raw_response_body_base64", "not-base64")])
def test_original_identity_response_integrity_is_checked_after_resealing(tmp_path, field, value):
    fixture = _fixture(tmp_path)
    fixture[3][NAMES[1]]["rows"][0][field] = value
    with pytest.raises(QT200IdentityEvidenceError):
        reconcile_qt200_identity_source_v1(**_resign_source(fixture, NAMES[1]))


def test_original_identity_missing_event_response_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    fixture[3][NAMES[1]]["rows"] = []
    with pytest.raises(QT200IdentityEvidenceError, match="identifier inventory"):
        reconcile_qt200_identity_source_v1(**_resign_source(fixture, NAMES[1]))


def test_original_identity_duplicate_event_response_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    fixture[3][NAMES[1]]["rows"] *= 2
    with pytest.raises(QT200IdentityEvidenceError, match="identifiers missing, additional, duplicated"):
        reconcile_qt200_identity_source_v1(**_resign_source(fixture, NAMES[1]))


@pytest.mark.parametrize("name", ["reference-records.jsonl", "ticker-event-captures.jsonl", "ticker-change-events.jsonl"])
@pytest.mark.parametrize("change", ["missing", "duplicate", "modified"])
def test_selected_identity_inventory_cannot_be_changed_with_new_support_hashes(tmp_path, name, change):
    fixture = _fixture(tmp_path)
    rows = [json.loads(line) for line in fixture[1][name].splitlines()]
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    else:
        rows[0]["provenance"]["result_index"] = 999
    with pytest.raises(QT200IdentityEvidenceError, match="selected support inventory/provenance"):
        reconcile_qt200_identity_source_v1(**_resign_support(fixture, name, rows))


def test_selected_reference_cannot_claim_another_qt200_target(tmp_path):
    fixture = _fixture(tmp_path)
    rows = [json.loads(line) for line in fixture[1]["reference-records.jsonl"].splitlines()]
    rows[0]["qt200_tickers"] = ["BBB"]
    with pytest.raises(QT200IdentityEvidenceError, match="inventory/provenance"):
        reconcile_qt200_identity_source_v1(**_resign_support(fixture, "reference-records.jsonl", rows))


def test_alias_assignment_and_candidate_only_flags_are_checked(tmp_path):
    fixture = _fixture(tmp_path)
    aliases = json.loads(fixture[1]["historical-aliases.json"])
    aliases["records"][0]["qt200_ticker"] = "BBB"
    with pytest.raises(QT200IdentityEvidenceError, match="historical-aliases"):
        reconcile_qt200_identity_source_v1(**_resign_support(fixture, "historical-aliases.json", aliases))


def test_original_reference_pagination_must_close(tmp_path):
    fixture = _fixture(tmp_path)
    fixture[3][NAMES[0]]["captures"][0]["pages"][0]["next_url"] = "https://api.massive.com/v3/reference/tickers?cursor=next"
    with pytest.raises(QT200IdentityEvidenceError):
        reconcile_qt200_identity_source_v1(**_resign_source(fixture, NAMES[0]))


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "oversize"])
def test_original_identity_source_fd_contract(tmp_path, kind):
    args, _, _, _ = _fixture(tmp_path)
    path = args["source_root"] / NAMES[0]
    if kind == "symlink":
        moved = tmp_path / "moved-reference.json"
        path.rename(moved)
        path.symlink_to(moved)
    elif kind == "hardlink":
        os.link(path, tmp_path / "extra-link.json")
    else:
        with path.open("r+b") as stream:
            stream.truncate(MAX_FILE_BYTES+1)
    with pytest.raises(QT200IdentityEvidenceError):
        reconcile_qt200_identity_source_v1(**args)


def test_original_identity_missing_capture_binding_is_rejected(tmp_path):
    args, _, _, _ = _fixture(tmp_path)
    args["expected_capture_sha256"].pop(NAMES[2])
    with pytest.raises(QT200IdentityEvidenceError, match="exactly three"):
        reconcile_qt200_identity_source_v1(**args)


def test_original_set_order_may_permute_target_blocks_within_one_response(tmp_path):
    fixture = _fixture(tmp_path, multi_target=True)
    rows = [json.loads(line) for line in fixture[1]["ticker-change-events.jsonl"].splitlines()]
    result = reconcile_qt200_identity_source_v1(**_resign_support(
        fixture, "ticker-change-events.jsonl", rows[2:]+rows[:2]))
    assert result["selected_ticker_change_count"] == 4
    assert result["selected_event_capture_count"] == 1
    assert result["selected_alias_count"] == 0
    assert result["historical_identity_qualified"] is False


@pytest.mark.parametrize("indices", [[1, 0, 2, 3], [0, 2, 1, 3]])
def test_original_set_order_does_not_relax_per_target_event_order_or_contiguity(tmp_path, indices):
    fixture = _fixture(tmp_path, multi_target=True)
    rows = [json.loads(line) for line in fixture[1]["ticker-change-events.jsonl"].splitlines()]
    with pytest.raises(QT200IdentityEvidenceError):
        reconcile_qt200_identity_source_v1(**_resign_support(
            fixture, "ticker-change-events.jsonl", [rows[i] for i in indices]))


def test_reference_cross_capture_chronology_is_checked(tmp_path):
    fixture = _fixture(tmp_path)
    page = fixture[3][NAMES[0]]["captures"][1]["pages"][0]
    page["requested_at_ms"] = STAMP
    with pytest.raises(QT200IdentityEvidenceError, match="cross-capture chronology"):
        reconcile_qt200_identity_source_v1(**_resign_source(fixture, NAMES[0]))
