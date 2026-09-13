"""Lossless packed originals, metadata lookup, corruption and CUDA parity."""

import base64
from dataclasses import asdict, replace
import gzip
from hashlib import sha256
import json
import os

import pytest
import torch

from raw_second_fixture import START, configure, make_catalog, response
from rl_quant.datasets import massive_raw_second_packed_v1 as packed
from rl_quant.datasets.massive_raw_seconds_v1 import (
    CapturedSecondPage, RawSecondCatalog, RawSecondContract, RawSecondWindowRef,
    SecondPartitionSet, SecondQuery, publish_second_capture,
    resolve_second_interval, second_capture_ref_from_dict,
)

pytestmark = pytest.mark.lsf_gpu


@pytest.fixture(autouse=True, scope="module")
def assigned_gpu():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    configure()


def _row(stamp, price=100):
    return dict(t=stamp, o=price, h=price + 1, l=price - 1, c=price + 0.25, v=1234.5)


def _page(query, rows, *, received=None, next_url=None):
    return CapturedSecondPage(query.url, received or query.end_ms + 5000,
                              response(query, rows, next_url=next_url))


def _pack(root, captures):
    with packed.SecondPackWriter(root, provenance={"synthetic_test_only": True}) as writer:
        for query, pages in captures:
            writer.begin_query(query)
            for page in pages:
                writer.append_page(page)
            writer.finish_query()
        result = writer.finalize()
    return tuple(packed.PackedSecondCaptureRef.from_dict(r) for r in result["captures"]), result


def _inventory(root):
    return {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def _rewrite(path, body):
    path.chmod(0o600)
    path.write_bytes(body)


def test_original_gzip_json_receipt_and_native_identity_are_preserved(tmp_path):
    query = SecondQuery("BRK.B", START, START + 7000)
    page = _page(query, [_row(START)])
    original = publish_second_capture(tmp_path / "loose", query, (page,))
    zipped = gzip.compress(page.body, compresslevel=1, mtime=12345)
    receipt = b'{ "source": "synthetic receipt", "request_id": "original" }\n'
    with packed.SecondPackWriter(tmp_path / "packed") as writer:
        writer.begin_query(query)
        writer.append_page(page, compressed_body=zipped, source_receipt=receipt)
        assert not (writer.root / packed.INDEX_NAME).exists()
        assert writer.finish_query() == original.manifest_sha256
        proof = writer.finalize()
    ref = packed.PackedSecondCaptureRef.from_dict(proof["captures"][0])
    assert ref.load() == original.load() == (query, (page,))
    assert ref.query() == query
    assert set(p.name for p in (tmp_path / "packed").iterdir()) == {packed.INDEX_NAME, packed.PACK_NAME}
    index = json.loads((tmp_path / "packed" / packed.INDEX_NAME).read_bytes())
    descriptor = index["captures"][0]["pages"][0]
    payload = (tmp_path / "packed" / packed.PACK_NAME).read_bytes()
    assert payload[descriptor["offset"]:descriptor["offset"] + descriptor["gzip_bytes"]] == zipped
    header = json.loads(payload[descriptor["frame_offset"] + 4:descriptor["offset"]])
    assert base64.b64decode(header["source_receipt_base64"]) == receipt
    before = _inventory(tmp_path)
    assert packed.verify_packed_second_store(root=tmp_path / "packed", index_sha256=ref.index_sha256) == proof
    assert _inventory(tmp_path) == before
    assert not proof["training_ready"] and not proof["point_in_time_qualified"]


def test_packed_catalog_preserves_all_raw_values_masks_clocks_and_identity(tmp_path):
    original = make_catalog(tmp_path / "loose")
    refs, _ = _pack(tmp_path / "packed", tuple(ref.load() for ref in original.windows[0].captures))
    catalog = RawSecondCatalog(tuple(replace(w, captures=refs) for w in original.windows))
    assert catalog.identity == original.identity
    assert all(second_capture_ref_from_dict(ref.to_dict()) == ref for ref in refs)
    assert second_capture_ref_from_dict(asdict(original.windows[0].captures[0])) == original.windows[0].captures[0]
    for index in range(len(catalog.windows)):
        first, second = original.load(index, device="cuda:0"), catalog.load(index, device="cuda:0")
        for name in ("raw_ohlcv", "observed_mask", "known_mask", "padding_mask", "second_timestamp_ms",
                     "available_at_ms", "decision_timestamp_ms"):
            assert torch.equal(getattr(first, name), getattr(second, name))
        assert second.asset_ids == first.asset_ids
        assert second.raw_ohlcv.dtype == torch.float32
        assert not bool(second.observed_mask[:, :, 10].any())
        assert bool(second.known_mask[:, :, 10].all())


@pytest.mark.parametrize("extra", [{"scaler": "forbidden"}, {"schema": "other"}])
def test_reference_restore_rejects_extra_storage_or_feature_fields(tmp_path, extra):
    row = dict(path=str(tmp_path), manifest_sha256="0" * 64, index_sha256="1" * 64)
    with pytest.raises(ValueError, match="reference fields"):
        second_capture_ref_from_dict(row | extra)


def test_literal_historical_ticker_is_not_rewritten_or_identity_qualified(tmp_path):
    query = SecondQuery("FB", START, START + 7000)
    refs, proof = _pack(tmp_path / "packed", ((query, (_page(query, [_row(START)]),)),))
    assert refs[0].load()[0].ticker == "FB"
    assert json.loads(refs[0].load()[1][0].body)["ticker"] == "FB"
    assert not proof["training_ready"] and not proof["point_in_time_qualified"]


@pytest.mark.parametrize("damage", ["index", "header", "gzip", "truncated", "trailing", "extra", "symlink", "hardlink"])
def test_packed_corruption_and_substitution_fail_closed(tmp_path, damage):
    query = SecondQuery("AAPL", START, START + 7000)
    refs, _ = _pack(tmp_path / "packed", ((query, (_page(query, [_row(START)]),)),))
    root, ref = tmp_path / "packed", refs[0]
    path = root / packed.PACK_NAME
    index = json.loads((root / packed.INDEX_NAME).read_bytes())
    if damage == "index":
        _rewrite(root / packed.INDEX_NAME, (root / packed.INDEX_NAME).read_bytes() + b" ")
    elif damage in ("header", "gzip"):
        page = index["captures"][0]["pages"][0]
        offset = page["frame_offset"] + 4 if damage == "header" else page["offset"] + 5
        body = bytearray(path.read_bytes())
        body[offset] ^= 1
        _rewrite(path, bytes(body))
    elif damage == "truncated":
        _rewrite(path, path.read_bytes()[:-1])
    elif damage == "trailing":
        _rewrite(path, path.read_bytes() + b"extra")
    elif damage == "extra":
        (root / "scaler.json").write_bytes(b"{}")
    elif damage == "symlink":
        moved = tmp_path / "moved-payload"
        path.rename(moved)
        path.symlink_to(moved)
    else:
        os.link(path, tmp_path / "other-link")
    with pytest.raises((ValueError, OSError)):
        ref.load()


@pytest.mark.parametrize("field,value", [("offset", 1), ("raw_bytes", packed.MAX_PAGE_BYTES + 1),
                                         ("header_bytes", packed.MAX_HEADER_BYTES + 1)])
def test_index_bounds_are_checked_before_decompression(tmp_path, field, value):
    query = SecondQuery("AAPL", START, START + 7000)
    refs, _ = _pack(tmp_path / "packed", ((query, (_page(query, [_row(START)]),)),))
    path = tmp_path / "packed" / packed.INDEX_NAME
    index = json.loads(path.read_bytes())
    index["captures"][0]["pages"][0][field] = value
    body = packed._canonical(index)
    _rewrite(path, body)
    malformed = replace(refs[0], index_sha256=sha256(body).hexdigest())
    with pytest.raises(ValueError, match="bounds|offsets"):
        malformed.query()


def test_bounded_inflation_rejects_bomb_and_extra_gzip_member():
    body = b"x" * 100_000
    with pytest.raises(ValueError, match="decoded size"):
        packed._inflate(gzip.compress(body), 1, sha256(b"x").hexdigest())
    with pytest.raises(ValueError, match="decoded size"):
        packed._inflate(gzip.compress(b"a") + gzip.compress(b"b"), 1, sha256(b"a").hexdigest())


@pytest.mark.parametrize("failed_response", [True, False])
def test_failed_or_incomplete_queries_preserve_frames_without_commit_or_resume(tmp_path, failed_response):
    root = tmp_path / "packed"
    query = SecondQuery("AAPL", START, START + 7000)
    page = (_page(query, [_row(START)], next_url=query.url + "&cursor=next") if not failed_response else
            CapturedSecondPage(query.url, START + 9000, b'{"status":"ERROR","message":"synthetic failure"}'))
    with packed.SecondPackWriter(root) as writer:
        writer.begin_query(query)
        writer.append_page(page)
        size = (root / packed.PACK_NAME).stat().st_size
        with pytest.raises(ValueError, match="pagination incomplete|failed response"):
            writer.finish_query()
        with pytest.raises(ValueError, match="failed packed writer"):
            writer.finalize()
    assert (root / packed.PACK_NAME).stat().st_size == size > len(packed.MAGIC)
    assert not (root / packed.INDEX_NAME).exists()
    with pytest.raises(FileExistsError):
        packed.SecondPackWriter(root)


def test_append_error_poisons_writer_and_never_commits_an_omitted_page(tmp_path):
    query = SecondQuery("AAPL", START, START + 7000)
    page = _page(query, [_row(START)])
    with packed.SecondPackWriter(tmp_path / "packed") as writer:
        writer.begin_query(query)
        writer.append_page(page)
        with pytest.raises(ValueError, match="decoded hash|decoded size"):
            writer.append_page(page, compressed_body=gzip.compress(b"not the source"))
        with pytest.raises(ValueError, match="failed packed writer"):
            writer.finish_query()
    assert not (tmp_path / "packed" / packed.INDEX_NAME).exists()


def test_capture_and_page_count_bounds_are_explicit(tmp_path):
    with packed.SecondPackWriter(tmp_path / "captures") as writer:
        for i in range(packed.MAX_CAPTURES):
            query = SecondQuery("AAPL", START + i * 8000, START + i * 8000 + 7000)
            writer.begin_query(query)
            writer.append_page(_page(query, []))
            writer.finish_query()
        with pytest.raises(ValueError, match="bounded query"):
            writer.begin_query(SecondQuery("AAPL", START + 24 * 8000, START + 25 * 8000 - 1000))
    query = SecondQuery("AAPL", START, START + 7000)
    with packed.SecondPackWriter(tmp_path / "pages") as writer:
        writer.begin_query(query)
        for _ in range(packed.MAX_PAGES_PER_CAPTURE):
            writer.append_page(_page(query, []))
        with pytest.raises(ValueError, match="bounded query"):
            writer.append_page(_page(query, []))


def test_metadata_interval_lookup_does_not_decompress_unrelated_prices(tmp_path):
    first = SecondQuery("AAPL", START, START + 7000)
    later = SecondQuery("AAPL", START + 8000, START + 15000)
    refs, proof = _pack(tmp_path / "packed", tuple((q, (_page(q, [_row(q.start_ms)]),)) for q in (first, later)))
    root = tmp_path / "packed"
    index = json.loads((root / packed.INDEX_NAME).read_bytes())
    body = bytearray((root / packed.PACK_NAME).read_bytes())
    body[index["captures"][1]["pages"][0]["offset"] + 5] ^= 1
    _rewrite(root / packed.PACK_NAME, bytes(body))
    # Selected reads authenticate selected frames; full verification detects
    # corruption elsewhere. Merely inspecting interval metadata opens no page.
    assert refs[1].query() == later
    assert resolve_second_interval(SecondPartitionSet(refs), START, START + 8000).rows[0][0] == START
    with pytest.raises(ValueError, match="gzip page changed"):
        refs[1].load()
    with pytest.raises(ValueError, match="Whole packed payload"):
        packed.verify_packed_second_store(root=root, index_sha256=proof["index_sha256"])


@pytest.mark.parametrize("conflict", ["price", "empty", "ticker"])
def test_mixed_loose_and_packed_conflicting_partitions_still_fail(tmp_path, conflict):
    query = SecondQuery("AAPL", START, START + 7000)
    loose = publish_second_capture(tmp_path / "loose", query, (_page(query, [_row(START)]),))
    other = replace(query, ticker="BRK.B") if conflict == "ticker" else query
    refs, _ = _pack(tmp_path / "packed", ((other, (_page(other, [] if conflict == "empty" else [_row(START, 101)]),)),))
    with pytest.raises(ValueError, match="Conflicting|Mixed ticker"):
        resolve_second_interval(SecondPartitionSet((loose, refs[0])), START, START + 8000)


def test_packed_known_empty_unknown_gap_and_delayed_padding_are_distinct(tmp_path):
    first, later = SecondQuery("AAPL", START, START + 7000), SecondQuery("AAPL", START + 9000, START + 15000)
    refs, _ = _pack(tmp_path / "packed", tuple((q, (_page(q, [_row(q.start_ms)]),)) for q in (first, later)))
    contract = RawSecondContract("developer-delayed-assumption", 900000)
    early = RawSecondWindowRef((refs[0],), ("issue-apple",), START, 8, START + 904000, contract)
    observation = RawSecondCatalog((early,)).load(0, device="cuda:0")
    assert bool(observation.observed_mask[0, 0, 0])
    assert bool(observation.known_mask.all())
    assert not bool(observation.observed_mask[0, 0, 1:4].any())
    assert not bool(observation.padding_mask[0, 0, :4].any())
    assert bool(observation.padding_mask[0, 0, 4:].all())
    gap = replace(early, captures=(SecondPartitionSet(refs),), seconds=16, decision_ms=START + 916000)
    with pytest.raises(ValueError, match="Unknown coverage"):
        RawSecondCatalog((gap,)).load(0, device="cuda:0")


def test_verified_transport_repacking_keeps_originals_and_reuses_original_gzip(tmp_path):
    from test_raw_second_partitions_v1 import _bounded_handoff

    query = SecondQuery("AAPL", START, START + 7000)
    _bounded_handoff(tmp_path / "input", (query,), {query.url: [_row(START)]})
    source = tmp_path / "input/http"
    before = _inventory(source)
    proof = packed.pack_verified_second_capture(root=source, plan_sha256=before["plan.json"],
        completion_sha256=before["COMPLETE.json"], output=tmp_path / "packed")
    assert _inventory(source) == before
    assert proof["capture_count"] == proof["page_count"] == 1
    assert not proof["original_inputs_deleted"]
    ref = packed.PackedSecondCaptureRef.from_dict(proof["captures"][0])
    index = json.loads((tmp_path / "packed" / packed.INDEX_NAME).read_bytes())
    page = index["captures"][0]["pages"][0]
    original_gzip = next(source.glob("*/page-0000.json.gz")).read_bytes()
    assert page["gzip_sha256"] == sha256(original_gzip).hexdigest()
    assert ref.load()[1][0].body == gzip.decompress(original_gzip)
    assert proof["source_provenance"]["provider_tickers_rewritten"] is False
