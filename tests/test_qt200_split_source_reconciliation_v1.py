from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pytest

from rl_quant.data_sources.massive.economic_provider_capture_v8 import (
    capture_massive_economic_rest_surface_for_test_v8,
)
from rl_quant.data_sources.massive.qt200_identity_evidence_v1 import (
    QT200IdentityEvidenceError,
    reconcile_qt200_split_source_v1,
)
from rl_quant.data_sources.massive.source_receipts import publish_massive_source_object


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha(body):
    return hashlib.sha256(body).hexdigest()


def _split(ticker="AAA", event_id="split-a"):
    return dict(ticker=ticker, id=event_id, execution_date="2022-01-03",
                split_from=1.0, split_to=2.0)


def _publish_support(root, files, sources):
    inventory = [dict(relative_path=name, bytes=len(body), sha256=_sha(body))
                 for name, body in files.items()]
    manifest = dict(files=copy.deepcopy(inventory), sources=sources,
                    universe_sha256=_sha(files["qt200-universe-spec.json"]),
                    requested_economic_coverage_start="2022-01-01",
                    requested_economic_coverage_end="2022-12-31")
    manifest_bytes = _bytes(manifest)
    inventory.append(dict(relative_path="manifest.json", bytes=len(manifest_bytes),
                          sha256=_sha(manifest_bytes)))
    inventory.sort(key=lambda item: item["relative_path"])
    complete = _bytes(dict(schema="quanttrade-qt200-support-extraction-complete-v1",
        status="COMPLETE_SUPPORT_DATA_EXTRACTION_ONLY", inventory=inventory,
        inventory_sha256=_sha(_bytes(inventory)), manifest_sha256=_sha(manifest_bytes),
        inventoried_file_count_excluding_complete=len(inventory),
        inventoried_bytes_excluding_complete=sum(row["bytes"] for row in inventory)))
    for name, body in {**files, "manifest.json": manifest_bytes, "COMPLETE.json": complete}.items():
        (root / name).write_bytes(body)
    return _sha(complete)


def _fixture(tmp_path, *, pages=None, surface="massive-splits-v1", alias=False):
    source = tmp_path / "source"
    source.mkdir()
    pages = pages if pages is not None else [[_split(), _split("OUT", "outside")], [_split("BBB", "split-b")]]
    kind = "splits" if surface == "massive-splits-v1" else "dividends"
    bodies = [_bytes(dict(status="OK", results=records, **(
        {"next_url": f"https://api.massive.com/stocks/v1/{kind}?cursor=p{i+1}"}
        if i+1 < len(pages) else {}))) for i, records in enumerate(pages)]
    capture = capture_massive_economic_rest_surface_for_test_v8(
        root=source, surface_id=surface, coverage_start_date="2022-01-01",
        coverage_end_date="2022-12-31", raw_page_bodies=bodies,
        requested_at_ms=1_800_000_000_000, completed_at_ms=1_800_000_000_100,
        entitlement_receipt_sha256="a"*64, capture_id="split-provenance-fixture")
    relative = capture.loaded_source.payload_relative_path
    original = source / relative
    name = original.name
    tickers = ["AAA", "BBB"] + [f"T{i:03d}" for i in range(198)]
    targets = {t: [t] for t in tickers}
    if alias:
        targets["OLD"] = ["AAA"]
    selected = []
    for page in capture.pages:
        for index, record in enumerate(page.parsed_body()["results"]):
            if record["ticker"] not in targets:
                continue
            selected.append(dict(provider_record=record,
                candidate_qt200_tickers=targets[record["ticker"]],
                corporate_action_permanent_identity_verified=False,
                historical_available_at=None, point_in_time_complete=False,
                automatic_adjustment_or_stitching_authorized=False,
                provenance=dict(source_file=name, source_sha256=capture.loaded_source.receipt.physical_sha256,
                    upstream_receipt_sha256=capture.loaded_source.receipt.receipt_sha256,
                    capture_index=None, page_index=page.page_index, result_index=index,
                    provider_request_id=page.provider_request_id,
                    raw_response_body_sha256=page.raw_response_body_sha256,
                    captured_at_ms=page.completed_at_ms,
                    captured_at_utc=datetime.fromtimestamp(page.completed_at_ms/1000, timezone.utc).isoformat(),
                    historical_available_at=None)))
    provenance = dict(source_file="identity-capture.json", source_sha256="b"*64,
                      upstream_receipt_sha256="c"*64, raw_response_body_sha256="d"*64,
                      captured_at_ms=1_800_000_000_200)
    refs = [dict(provider_record=dict(ticker=t, active=True, composite_figi="FIGI-"+t),
                 provenance=provenance) for t in ("AAA", "BBB")]
    events = [dict(qt200_tickers=["AAA"], request_identifier="FIGI-AAA", provenance=provenance,
        response_metadata=dict(response_status=200), provider_response=dict(status="OK",
            results=dict(composite_figi="FIGI-AAA", events=[dict(date="2020-01-01", type="ticker_change",
                ticker_change=dict(ticker="OLD"))])))] if alias else []
    support = tmp_path / "support"
    support.mkdir()
    files = {"qt200-universe-spec.json": _bytes(dict(ticker_count=200, tickers=tickers)),
             "reference-records.jsonl": b"".join(_bytes(r)+b"\n" for r in refs),
             "ticker-event-captures.jsonl": b"".join(_bytes(r)+b"\n" for r in events),
             "splits.jsonl": b"".join(_bytes(r)+b"\n" for r in selected), "dividends.jsonl": b""}
    sources = [dict(source_file=name, source_sha256=capture.loaded_source.receipt.physical_sha256,
                    source_bytes=original.stat().st_size,
                    upstream_receipt_sha256=capture.loaded_source.receipt.receipt_sha256),
               dict(source_file="identity-capture.json", source_sha256="b"*64,
                    upstream_receipt_sha256="c"*64)]
    args = dict(source_root=source, relative_payload_path=relative,
        expected_payload_sha256=_sha(original.read_bytes()),
        expected_receipt_file_sha256=_sha(Path(str(original)+".receipt.json").read_bytes()),
        expected_commit_file_sha256=_sha(Path(str(original)+".commit.json").read_bytes()),
        support_root=support, expected_support_completion_sha256=_publish_support(support, files, sources),
        expected_universe_sha256=_sha(files["qt200-universe-spec.json"]))
    return args, files, sources, selected


def _resign_selected(fixture, rows):
    args, files, sources, _ = fixture
    files["splits.jsonl"] = b"".join(_bytes(row)+b"\n" for row in rows)
    args["expected_support_completion_sha256"] = _publish_support(args["support_root"], files, sources)
    return args


def _inventory(root):
    return {str(p.relative_to(root)): (_sha(p.read_bytes()), p.stat().st_mode, p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def test_native_split_source_reconciliation_is_complete_read_only_and_nonauthorizing(tmp_path):
    args, _, _, _ = _fixture(tmp_path)
    before = _inventory(tmp_path)
    result = reconcile_qt200_split_source_v1(**args)
    assert _inventory(tmp_path) == before
    assert result["selected_split_count"] == 2
    assert result["original_source_result_count"] == 3
    assert result["page_count"] == 2
    assert result["split_original_provider_raw_bodies_reauthenticated"] is True
    assert result["exact_selected_position_inventory_verified"] is True
    for name in ("fixed_runtime_captured", "all_support_original_provider_raw_bodies_reauthenticated",
                 "historical_identity_qualified", "historical_availability_qualified",
                 "economic_accounting_qualified", "point_in_time_qualified",
                 "training_ready_for_adaptive_v5", "source_writes"):
        assert result[name] is False
    assert result["capture_kind"] == "synthetic-test-response-v8"
    assert result["provider_calls"] == result["native_security_ids_emitted"] == 0
    assert result["receipt_sha256"] == _sha(_bytes({k: v for k, v in result.items() if k != "receipt_sha256"}))
    assert reconcile_qt200_split_source_v1(**args) == result


def test_identical_original_rows_at_different_positions_are_not_deduplicated(tmp_path):
    args, _, _, _ = _fixture(tmp_path, pages=[[_split(), _split()]])
    result = reconcile_qt200_split_source_v1(**args)
    assert result["selected_split_count"] == 2
    assert [r["result_index"] for r in result["reconciled_splits"]] == [0, 1]


def test_captured_same_figi_alias_is_retained_without_identity_promotion(tmp_path):
    args, _, _, _ = _fixture(tmp_path, pages=[[_split("OLD")]], alias=True)
    result = reconcile_qt200_split_source_v1(**args)
    row = result["reconciled_splits"][0]
    assert row["candidate_qt200_tickers"] == ["AAA"]
    assert row["identity_qualified"] is False
    assert row["historical_available_at_ms"] is None


@pytest.mark.parametrize("suffix", ["", ".receipt.json", ".commit.json"])
def test_changed_transaction_bytes_reject_original_expected_hash(tmp_path, suffix):
    args, _, _, _ = _fixture(tmp_path)
    path = Path(str(args["source_root"] / args["relative_payload_path"])+suffix)
    path.chmod(0o600)
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(QT200IdentityEvidenceError, match="physical hash"):
        reconcile_qt200_split_source_v1(**args)


def test_truncated_original_payload_rejects_before_native_decode(tmp_path):
    args, _, _, _ = _fixture(tmp_path)
    path = args["source_root"] / args["relative_payload_path"]
    path.chmod(0o600)
    path.write_bytes(path.read_bytes()[:50])
    with pytest.raises(QT200IdentityEvidenceError, match="physical hash"):
        reconcile_qt200_split_source_v1(**args)


@pytest.mark.parametrize("field,value", [
    ("page_index", 99), ("result_index", 99), ("page_index", True),
    ("captured_at_ms", 1), ("captured_at_utc", "2022-01-03T00:00:00+00:00"),
    ("provider_request_id", "wrong"), ("raw_response_body_sha256", "f"*64),
])
def test_resealed_support_cannot_change_original_page_provenance(tmp_path, field, value):
    fixture = _fixture(tmp_path)
    rows = fixture[3]
    rows[0]["provenance"][field] = value
    with pytest.raises(QT200IdentityEvidenceError, match="provenance"):
        reconcile_qt200_split_source_v1(**_resign_selected(fixture, rows))


def test_resealed_support_cannot_change_selected_provider_values(tmp_path):
    fixture = _fixture(tmp_path)
    rows = fixture[3]
    rows[0]["provider_record"]["split_to"] = 1000
    with pytest.raises(QT200IdentityEvidenceError, match="inventory mismatch"):
        reconcile_qt200_split_source_v1(**_resign_selected(fixture, rows))


def test_missing_selected_row_is_rejected_with_valid_support_hashes(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(QT200IdentityEvidenceError, match="Missing selected"):
        reconcile_qt200_split_source_v1(**_resign_selected(fixture, fixture[3][:-1]))


def test_duplicate_selected_provenance_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(QT200IdentityEvidenceError, match="Duplicate selected"):
        reconcile_qt200_split_source_v1(**_resign_selected(fixture, fixture[3]+[fixture[3][0]]))


def test_extra_nonselected_position_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    extra = copy.deepcopy(fixture[3][0])
    extra["provenance"]["result_index"] = 1
    with pytest.raises(QT200IdentityEvidenceError, match="Additional selected"):
        reconcile_qt200_split_source_v1(**_resign_selected(fixture, fixture[3]+[extra]))


def test_wrong_candidate_assignment_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    fixture[3][0]["candidate_qt200_tickers"] = ["BBB"]
    with pytest.raises(QT200IdentityEvidenceError, match="inventory mismatch"):
        reconcile_qt200_split_source_v1(**_resign_selected(fixture, fixture[3]))


def test_dividend_transaction_cannot_be_used_as_split_evidence(tmp_path):
    args, _, _, _ = _fixture(tmp_path, surface="massive-dividends-v1")
    with pytest.raises(QT200IdentityEvidenceError, match="native split transaction"):
        reconcile_qt200_split_source_v1(**args)


def test_source_sidecar_cannot_be_symlinked(tmp_path):
    args, _, _, _ = _fixture(tmp_path)
    path = Path(str(args["source_root"] / args["relative_payload_path"])+".receipt.json")
    original = tmp_path / "moved-receipt.json"
    path.rename(original)
    path.symlink_to(original)
    with pytest.raises(QT200IdentityEvidenceError, match="Symlink"):
        reconcile_qt200_split_source_v1(**args)


def test_native_raw_page_hash_is_checked_even_with_republished_transaction(tmp_path):
    args, _, _, _ = _fixture(tmp_path)
    original = args["source_root"] / args["relative_payload_path"]
    document = json.loads(original.read_bytes())
    document["pages"][0]["raw_response_body_sha256"] = "e"*64
    source_receipt = json.loads(Path(str(original)+".receipt.json").read_bytes())
    source_commit = json.loads(Path(str(original)+".commit.json").read_bytes())
    replacement = tmp_path / "republished"
    replacement.mkdir()
    publish_massive_source_object(stream=BytesIO(_bytes(document)+b"\n"), root=replacement,
        relative_payload_path=args["relative_payload_path"],
        dataset_id=source_receipt["dataset_id"], source_object_key=source_receipt["source_object_key"],
        requested_at_ms=source_receipt["requested_at_ms"], downloaded_at_ms=source_receipt["downloaded_at_ms"],
        schema_sha256=source_receipt["schema_sha256"],
        entitlement_receipt_sha256=source_receipt["entitlement_receipt_sha256"],
        committed_at_ms=source_commit["committed_at_ms"], request_id=source_receipt["request_id"])
    args["source_root"] = replacement
    changed = replacement / args["relative_payload_path"]
    for key, suffix in (("expected_payload_sha256", ""), ("expected_receipt_file_sha256", ".receipt.json"),
                        ("expected_commit_file_sha256", ".commit.json")):
        args[key] = _sha(Path(str(changed)+suffix).read_bytes())
    with pytest.raises(QT200IdentityEvidenceError, match="Native split transaction/page replay"):
        reconcile_qt200_split_source_v1(**args)
