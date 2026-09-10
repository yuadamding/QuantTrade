from __future__ import annotations

import copy
import json
from io import BytesIO
from pathlib import Path

import pytest

from rl_quant.data_sources.massive.economic_provider_capture_v8 import (
    capture_massive_economic_rest_surface_for_test_v8,
)
from rl_quant.data_sources.massive.qt200_identity_evidence_v1 import (
    MAX_DIVIDEND_SOURCE_BYTES,
    MAX_DIVIDEND_SOURCE_PAGES,
    MAX_DIVIDEND_SOURCE_RESULTS,
    MAX_FILE_BYTES,
    QT200IdentityEvidenceError,
    reconcile_qt200_dividend_source_v1,
)
from rl_quant.data_sources.massive.source_receipts import publish_massive_source_object
from test_qt200_split_source_reconciliation_v1 import (
    _bytes,
    _fixture as _native_source_fixture,
    _inventory,
    _publish_support,
    _sha,
)


def _dividend(ticker="AAA", event_id="dividend-a"):
    return dict(ticker=ticker, id=event_id, ex_dividend_date="2022-01-03",
                declaration_date="2021-12-15", pay_date="2022-01-20",
                record_date="2022-01-04", cash_amount=0.25, currency="USD",
                distribution_type="CD", frequency=4)


def _fixture(tmp_path, *, pages=None, surface="massive-dividends-v1", alias=False):
    """Use the same real native capture/publisher, not fabricated authorities."""
    if pages is None:
        pages = [[_dividend(), _dividend("OUT", "outside")], [_dividend("BBB", "dividend-b")]]
    args, files, sources, selected = _native_source_fixture(
        tmp_path, pages=pages, surface=surface, alias=alias)
    files["dividends.jsonl"], files["splits.jsonl"] = files["splits.jsonl"], b""
    args["expected_support_completion_sha256"] = _publish_support(args["support_root"], files, sources)
    return args, files, sources, selected


def _resign_selected(fixture, rows):
    args, files, sources, _ = fixture
    files["dividends.jsonl"] = b"".join(_bytes(row)+b"\n" for row in rows)
    args["expected_support_completion_sha256"] = _publish_support(args["support_root"], files, sources)
    return args


def test_native_dividend_reconciliation_is_exact_read_only_and_nonauthorizing(tmp_path):
    args, _, _, _ = _fixture(tmp_path)
    before = _inventory(tmp_path)
    result = reconcile_qt200_dividend_source_v1(**args)
    assert _inventory(tmp_path) == before
    assert result["selected_dividend_count"] == 2
    assert result["original_source_result_count"] == 3
    assert result["page_count"] == 2
    assert result["dividend_original_provider_raw_bodies_reauthenticated"] is True
    assert result["exact_selected_position_inventory_verified"] is True
    assert result["original_transaction_replayed"] is True
    assert result["capture_kind"] == "synthetic-test-response-v8"
    for name in ("fixed_runtime_captured", "all_support_original_provider_raw_bodies_reauthenticated",
                 "historical_identity_qualified", "historical_availability_qualified",
                 "economic_accounting_qualified", "point_in_time_qualified",
                 "training_ready_for_adaptive_v5", "source_writes"):
        assert result[name] is False
    assert result["provider_calls"] == result["native_security_ids_emitted"] == result["successor_joins_emitted"] == 0
    for row in result["reconciled_dividends"]:
        assert row["identity_qualified"] is False
        assert row["accounting_authorized"] is False
        assert row["historical_available_at_ms"] is None
        assert row["provenance"]["historical_available_at"] is None
    assert result["receipt_sha256"] == _sha(_bytes({k: v for k, v in result.items() if k != "receipt_sha256"}))
    assert reconcile_qt200_dividend_source_v1(**args) == result


def test_identical_original_dividends_at_distinct_positions_are_preserved(tmp_path):
    args, _, _, _ = _fixture(tmp_path, pages=[[_dividend(), _dividend()]])
    result = reconcile_qt200_dividend_source_v1(**args)
    assert result["selected_dividend_count"] == 2
    assert [r["result_index"] for r in result["reconciled_dividends"]] == [0, 1]


def test_dividend_same_figi_alias_remains_only_a_candidate(tmp_path):
    args, _, _, _ = _fixture(tmp_path, pages=[[_dividend("OLD")]], alias=True)
    result = reconcile_qt200_dividend_source_v1(**args)
    assert result["reconciled_dividends"][0]["candidate_qt200_tickers"] == ["AAA"]
    assert result["historical_identity_qualified"] is False


@pytest.mark.parametrize("suffix", ["", ".receipt.json", ".commit.json"])
def test_dividend_changed_original_transaction_bytes_are_rejected(tmp_path, suffix):
    args, _, _, _ = _fixture(tmp_path)
    path = Path(str(args["source_root"] / args["relative_payload_path"])+suffix)
    path.chmod(0o600)
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(QT200IdentityEvidenceError, match="physical hash"):
        reconcile_qt200_dividend_source_v1(**args)


def test_truncated_dividend_payload_is_rejected_before_native_decode(tmp_path):
    args, _, _, _ = _fixture(tmp_path)
    path = args["source_root"] / args["relative_payload_path"]
    path.chmod(0o600)
    path.write_bytes(path.read_bytes()[:50])
    with pytest.raises(QT200IdentityEvidenceError, match="physical hash"):
        reconcile_qt200_dividend_source_v1(**args)


@pytest.mark.parametrize("field,value", [
    ("page_index", 99), ("result_index", 99), ("page_index", True),
    ("captured_at_ms", 1), ("captured_at_utc", "2022-01-03T00:00:00+00:00"),
    ("provider_request_id", "wrong"), ("raw_response_body_sha256", "f"*64),
])
def test_resealed_dividend_support_cannot_change_original_provenance(tmp_path, field, value):
    fixture = _fixture(tmp_path)
    fixture[3][0]["provenance"][field] = value
    with pytest.raises(QT200IdentityEvidenceError):
        reconcile_qt200_dividend_source_v1(**_resign_selected(fixture, fixture[3]))


@pytest.mark.parametrize("field,value", [("cash_amount", 1000), ("pay_date", "2022-02-01")])
def test_resealed_dividend_support_cannot_change_provider_fields(tmp_path, field, value):
    fixture = _fixture(tmp_path)
    fixture[3][0]["provider_record"][field] = value
    with pytest.raises(QT200IdentityEvidenceError, match="inventory mismatch"):
        reconcile_qt200_dividend_source_v1(**_resign_selected(fixture, fixture[3]))


def test_missing_selected_dividend_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(QT200IdentityEvidenceError, match="Missing selected"):
        reconcile_qt200_dividend_source_v1(**_resign_selected(fixture, fixture[3][:-1]))


def test_duplicate_selected_dividend_provenance_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(QT200IdentityEvidenceError, match="Duplicate selected"):
        reconcile_qt200_dividend_source_v1(**_resign_selected(fixture, fixture[3]+[fixture[3][0]]))


def test_extra_nonselected_dividend_position_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    extra = copy.deepcopy(fixture[3][0])
    extra["provenance"]["result_index"] = 1
    with pytest.raises(QT200IdentityEvidenceError, match="Additional selected"):
        reconcile_qt200_dividend_source_v1(**_resign_selected(fixture, fixture[3]+[extra]))


def test_reordered_dividend_positions_are_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(QT200IdentityEvidenceError, match="inventory mismatch"):
        reconcile_qt200_dividend_source_v1(**_resign_selected(fixture, list(reversed(fixture[3]))))


def test_dividend_wrong_candidate_assignment_is_rejected(tmp_path):
    fixture = _fixture(tmp_path)
    fixture[3][0]["candidate_qt200_tickers"] = ["BBB"]
    with pytest.raises(QT200IdentityEvidenceError, match="inventory mismatch"):
        reconcile_qt200_dividend_source_v1(**_resign_selected(fixture, fixture[3]))


def test_split_transaction_cannot_be_used_as_dividend_evidence(tmp_path):
    args, _, _, _ = _fixture(tmp_path, surface="massive-splits-v1")
    with pytest.raises(QT200IdentityEvidenceError, match="native dividend transaction"):
        reconcile_qt200_dividend_source_v1(**args)


@pytest.mark.parametrize("flag", ["point_in_time_complete", "automatic_adjustment_or_stitching_authorized"])
def test_dividend_support_cannot_claim_economic_qualification(tmp_path, flag):
    fixture = _fixture(tmp_path)
    fixture[3][0][flag] = True
    with pytest.raises(QT200IdentityEvidenceError, match="nonqualification flags"):
        reconcile_qt200_dividend_source_v1(**_resign_selected(fixture, fixture[3]))


def test_native_dividend_raw_page_hash_is_checked_after_transaction_republication(tmp_path):
    args, _, _, _ = _fixture(tmp_path)
    original = args["source_root"] / args["relative_payload_path"]
    document = json.loads(original.read_bytes())
    document["pages"][0]["raw_response_body_sha256"] = "e"*64
    receipt = json.loads(Path(str(original)+".receipt.json").read_bytes())
    commit = json.loads(Path(str(original)+".commit.json").read_bytes())
    replacement = tmp_path / "republished"
    replacement.mkdir()
    publish_massive_source_object(stream=BytesIO(_bytes(document)+b"\n"), root=replacement,
        relative_payload_path=args["relative_payload_path"],
        dataset_id=receipt["dataset_id"], source_object_key=receipt["source_object_key"],
        requested_at_ms=receipt["requested_at_ms"], downloaded_at_ms=receipt["downloaded_at_ms"],
        schema_sha256=receipt["schema_sha256"], entitlement_receipt_sha256=receipt["entitlement_receipt_sha256"],
        committed_at_ms=commit["committed_at_ms"], request_id=receipt["request_id"])
    args["source_root"] = replacement
    changed = replacement / args["relative_payload_path"]
    for key, suffix in (("expected_payload_sha256", ""), ("expected_receipt_file_sha256", ".receipt.json"),
                        ("expected_commit_file_sha256", ".commit.json")):
        args[key] = _sha(Path(str(changed)+suffix).read_bytes())
    with pytest.raises(QT200IdentityEvidenceError, match="Native dividend transaction/page replay"):
        reconcile_qt200_dividend_source_v1(**args)


def test_original_dividend_larger_than_support_cap_is_natively_replayed(tmp_path):
    outside = _dividend("OUT", "large-unselected-provider-record")
    outside["provider_extra_text"] = "x" * (13 * 1024 * 1024)
    args, _, _, _ = _fixture(tmp_path, pages=[[_dividend(), outside]])
    assert MAX_FILE_BYTES < (args["source_root"] / args["relative_payload_path"]).stat().st_size < MAX_DIVIDEND_SOURCE_BYTES
    result = reconcile_qt200_dividend_source_v1(**args)
    assert result["selected_dividend_count"] == 1
    assert result["original_source_result_count"] == 2


def test_oversized_original_dividend_is_rejected_before_read_or_decode(tmp_path):
    args, _, _, _ = _fixture(tmp_path)
    path = args["source_root"] / args["relative_payload_path"]
    path.chmod(0o600)
    with path.open("r+b") as stream:
        stream.truncate(MAX_DIVIDEND_SOURCE_BYTES + 1)
    with pytest.raises(QT200IdentityEvidenceError, match="single-link bounded regular"):
        reconcile_qt200_dividend_source_v1(**args)


def test_larger_original_limit_does_not_widen_support_file_cap(tmp_path):
    args, _, _, _ = _fixture(tmp_path)
    path = args["support_root"] / "dividends.jsonl"
    with path.open("r+b") as stream:
        stream.truncate(MAX_FILE_BYTES + 1)
    with pytest.raises(QT200IdentityEvidenceError, match="owned bounded regular"):
        reconcile_qt200_dividend_source_v1(**args)


def test_larger_original_limit_does_not_widen_sidecar_cap(tmp_path):
    args, _, _, _ = _fixture(tmp_path)
    path = Path(str(args["source_root"] / args["relative_payload_path"])+".receipt.json")
    path.chmod(0o600)
    with path.open("r+b") as stream:
        stream.truncate(MAX_FILE_BYTES + 1)
    with pytest.raises(QT200IdentityEvidenceError, match="owned bounded regular"):
        reconcile_qt200_dividend_source_v1(**args)


def test_dividend_page_count_bound_uses_the_native_page_chain(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    count = MAX_DIVIDEND_SOURCE_PAGES + 1
    bodies = [_bytes(dict(status="OK", results=[_dividend("OUT")], **(
        {"next_url": f"https://api.massive.com/stocks/v1/dividends?cursor=p{i+1}"}
        if i+1 < count else {}))) for i in range(count)]
    # The shared small fixture has only a 100 ms capture interval. Use enough
    # chronological time for 129 real pages so THIS diagnostic cap is tested.
    capture = capture_massive_economic_rest_surface_for_test_v8(
        root=source, surface_id="massive-dividends-v1", coverage_start_date="2022-01-01",
        coverage_end_date="2022-12-31", raw_page_bodies=bodies,
        requested_at_ms=1_800_000_000_000, completed_at_ms=1_800_000_001_000,
        entitlement_receipt_sha256="a"*64, capture_id="dividend-page-limit-fixture")
    original = source / capture.loaded_source.payload_relative_path
    unopened_support = tmp_path / "unopened-support"
    args = dict(source_root=source, relative_payload_path=capture.loaded_source.payload_relative_path,
        expected_payload_sha256=_sha(original.read_bytes()),
        expected_receipt_file_sha256=_sha(Path(str(original)+".receipt.json").read_bytes()),
        expected_commit_file_sha256=_sha(Path(str(original)+".commit.json").read_bytes()),
        support_root=unopened_support, expected_support_completion_sha256="b"*64,
        expected_universe_sha256="c"*64)
    with pytest.raises(QT200IdentityEvidenceError, match="bounded nonauthorizing native dividend"):
        reconcile_qt200_dividend_source_v1(**args)
    assert not unopened_support.exists()


def test_dividend_result_count_bound_covers_unselected_provider_rows(tmp_path):
    pages = [[_dividend("OUT")]] * 2
    # Real native pages; the result bound is narrowed only for this guard test.
    args, _, _, _ = _fixture(tmp_path, pages=pages)
    assert MAX_DIVIDEND_SOURCE_RESULTS == 500_000
    from unittest.mock import patch
    with patch("rl_quant.data_sources.massive.qt200_identity_evidence_v1.MAX_DIVIDEND_SOURCE_RESULTS", 1):
        with pytest.raises(QT200IdentityEvidenceError, match="bounded nonauthorizing native dividend"):
            reconcile_qt200_dividend_source_v1(**args)


def test_original_dividend_payload_symlink_is_rejected(tmp_path):
    args, _, _, _ = _fixture(tmp_path)
    path = args["source_root"] / args["relative_payload_path"]
    relocated = tmp_path / "moved-original.json"
    path.rename(relocated)
    path.symlink_to(relocated)
    with pytest.raises(QT200IdentityEvidenceError, match="Symlink"):
        reconcile_qt200_dividend_source_v1(**args)
