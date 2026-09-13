"""LSF-routed controller regression; synthetic HTTP, no real acquisition."""

from dataclasses import replace
import gzip
from io import BytesIO
from itertools import islice
import time
from urllib.parse import urlsplit

import pytest
import torch

from rl_quant.data_sources.massive import raw_second_corpus_v1 as corpus
from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport


pytestmark = pytest.mark.lsf_gpu
START = 1483453800000


@pytest.fixture(scope="module", autouse=True)
def assigned_gpu():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1


def _write(path, value):
    return transport.write_once(path, transport.canonical(value))["sha256"]


def _identity(**kwargs):
    return {"fixed_slot_ticker": kwargs["fixed_slot_ticker"],
            "session_date": kwargs["session_date"], "training_ready": False}


def _registry(tmp_path):
    root = tmp_path / "registry"
    root.mkdir()
    intended = tmp_path / "intended.json"
    identity_sha = _write(intended, {"synthetic_controller_fixture": True})
    tickers = list(corpus.SYMBOLS)
    summary = {"start_date": "2017-01-03", "end_date": "2026-08-31",
        "ordered_universe_sha256": transport.digest(transport.canonical(tickers)),
        "calendar_versions": {"exchange-calendars": "4.13.2"}, "calendar_is_planning_only": True,
        "identity_evidence": {"sha256": identity_sha},
        "market_contract": {"market_fields": ["open", "high", "low", "close", "volume"],
            "adjusted": False, "timespan": "second", "multiplier": 1, "availability_delay_ms": 900000,
            "availability_assumption": "developer-delayed-assumption",
            "feature_engineering": False, "resampling": None, "raw_input_normalization": None}}
    rows = []
    for day, date in enumerate(("2017-01-03", "2017-01-04")):
        left, close = START + day * 86400000, START + day * 86400000 + 23400000
        rows.append({"session_date": date, "open_ms": left, "close_ms": close,
            "partitions": [{"start_ms": a, "end_ms": min(close, a + 3600000) - 1000}
                           for a in range(left, close, 3600000)]})
    contents = {"universe.json": transport.canonical({"tickers": tickers}),
        "summary.json": transport.canonical(summary),
        "sessions.jsonl": b"".join(transport.canonical(row) for row in rows),
        "reuse-queries.jsonl": b""}
    files = [transport.write_once(root / name, raw) for name, raw in contents.items()]
    terminal = _write(root / "COMPLETE.json", {"metadata_complete": True, "files": files})
    return corpus.CorpusRegistry(root, terminal), intended


def _network(monkeypatch, *, fail=False):
    calls = []

    class Response(BytesIO):
        status = 403 if fail else 200
        code = status
        headers = {"Content-Type": "application/json"}

        def __init__(self, body, url):
            super().__init__(body)
            self.url = url

        def geturl(self):
            return self.url

    class Opener:
        def open(self, request, timeout):
            calls.append(request.full_url)
            pieces = urlsplit(request.full_url).path.split("/")
            ticker, start = pieces[4], int(pieces[-2])
            body = {"status": "OK", "ticker": ticker, "adjusted": False,
                    "request_id": "synthetic-request", "resultsCount": 1,
                    "results": [{"t": start, "o": 100, "h": 101, "l": 99, "c": 100, "v": 10}]}
            return Response(transport.canonical(body), request.full_url)

    monkeypatch.setattr(transport.request, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(transport.time, "sleep", lambda _: None)
    return calls


def _capacity(**changes):
    result = {"observed_at_ns": time.time_ns(), "project_ceiling_bytes": 900000000000,
        "project_allocated_bytes": 693000000000, "filesystem_free_bytes": 1000000000000,
        "fileset_available_bytes": 300000000000, "filesystem_free_inodes": 1000000,
        "fileset_available_inodes": 1000000, **changes}
    return {**result, "receipt_sha256": transport.digest(transport.canonical(result))}


def _setup(tmp_path, monkeypatch, *, storage_mode=corpus.LEGACY_STORAGE, **limits):
    monkeypatch.setattr(corpus, "validate_fixed_slot_identity", _identity)
    registry, intended = _registry(tmp_path)
    limit = corpus.CorpusLimits(maximum_http_requests=2000, maximum_raw_response_bytes=1000000000,
        maximum_retained_bytes=1000000000, project_ceiling_bytes=900000000000, tranche_batches=1)
    limit = replace(limit, **limits)
    root, owner = tmp_path / "attempt", tmp_path / "owner"
    plan_sha = corpus.publish_corpus_plan(root=root, registry=registry, limits=limit,
                                         intended_issue_source=intended, owner=owner,
                                         storage_mode=storage_mode)
    batch = next(registry.batches())
    identities = tuple(corpus.CorpusIdentity(ticker, ticker,
        corpus.AliasIdentityRef("dated-ticker-wrapper", str(intended), "a" * 64))
        for ticker in batch["fixed_slot_tickers"])
    return root, plan_sha, identities, registry, intended, limit, owner


def _run(setup, **kwargs):
    root, plan_sha, identities, *_ = setup
    return corpus.run_corpus_batch(root=root, plan_sha256=plan_sha, identities=identities,
        api_key="synthetic-test-only", capacity_snapshot=_capacity(), **kwargs)


def test_registry_iterator_is_lazy_stable_hour_bounded_and_keeps_all_200(tmp_path):
    registry, _ = _registry(tmp_path)
    iterator = registry.batches()
    assert iter(iterator) is iterator
    first_hour = list(islice(iterator, 9))
    assert [t for b in first_hour for t in b["fixed_slot_tickers"]] == list(corpus.SYMBOLS)
    assert [len(b["fixed_slot_tickers"]) for b in first_hour] == [24] * 8 + [8]
    assert all(b["end_ms"] - b["start_ms"] == 3599000 for b in first_hour)
    assert next(registry.batches(63))["session_date"] == "2017-01-04"
    assert next(registry.batches(8)) == first_hour[8]


@pytest.mark.parametrize("limits", [{"maximum_http_requests": 191},
    {"maximum_raw_response_bytes": 127999999}, {"maximum_retained_bytes": 128000000}])
def test_global_bounds_stop_before_acquisition_or_reservation(tmp_path, monkeypatch, limits):
    setup = _setup(tmp_path, monkeypatch, **limits)
    calls = _network(monkeypatch)
    with pytest.raises(transport.ResearchCaptureError, match="Global corpus bound"):
        _run(setup)
    assert not calls and not list((setup[0] / "batches").iterdir())


@pytest.mark.parametrize("changes", [{"observed_at_ns": 0}, {"fileset_available_bytes": 1},
    {"filesystem_free_inodes": 1}, {"project_allocated_bytes": 900000000000}])
def test_fresh_capacity_stops_before_capture(tmp_path, monkeypatch, changes):
    root, plan, identities, *_ = _setup(tmp_path, monkeypatch)
    calls = _network(monkeypatch)
    with pytest.raises(transport.ResearchCaptureError, match="capacity admission"):
        corpus.run_corpus_batch(root=root, plan_sha256=plan, identities=identities,
            api_key="synthetic-test-only", capacity_snapshot=_capacity(**changes))
    assert not calls and not list((root / "batches").iterdir())


def test_missing_identity_and_consumed_failure_never_issue_a_second_request(tmp_path, monkeypatch):
    setup = _setup(tmp_path, monkeypatch)
    calls = _network(monkeypatch, fail=True)
    with pytest.raises(transport.ResearchCaptureError, match="Missing"):
        corpus.run_corpus_batch(root=setup[0], plan_sha256=setup[1], identities=(),
            api_key="synthetic-test-only", capacity_snapshot=_capacity())
    assert not calls
    with pytest.raises(transport.ResearchCaptureError, match="Capture incomplete"):
        _run(setup)
    assert len(calls) == 1
    with pytest.raises(transport.ResearchCaptureError, match="ambiguous"):
        _run(setup)
    assert len(calls) == 1
    started = corpus._json(setup[0] / "batches/batch-000000/STARTED.json")
    assert started["reservation"]["http_requests"] == 192 and not started["retry_allowed"]


def test_completed_prefix_is_replayed_without_http_then_corruption_fails_closed(tmp_path, monkeypatch):
    setup = _setup(tmp_path, monkeypatch)
    calls = _network(monkeypatch)
    result = _run(setup)
    assert result["actual"]["http_requests"] == 24 and result["batch_complete"]
    terminal = _run(setup)
    assert terminal["range_complete"] and len(calls) == 24
    page = next((setup[0] / "batches/batch-000000/capture").glob("second-*/page-0000.json.gz"))
    page.chmod(0o600)
    page.write_bytes(gzip.compress(b"{}", mtime=0))
    with pytest.raises((ValueError, transport.ResearchCaptureError)):
        _run(setup)
    assert len(calls) == 24


def test_modified_counter_cannot_release_reserved_global_budget(tmp_path, monkeypatch):
    setup = _setup(tmp_path, monkeypatch)
    calls = _network(monkeypatch)
    _run(setup)
    path = setup[0] / "batches/batch-000000/COMPLETE.json"
    body = corpus._json(path)
    body["actual"]["http_requests"] = 0
    body["consumed_after"]["http_requests"] = 0
    path.chmod(0o600)
    path.write_bytes(transport.canonical(body))
    with pytest.raises(transport.ResearchCaptureError, match="accounting"):
        _run(setup)
    assert len(calls) == 24


def test_second_successor_and_changed_plan_are_rejected(tmp_path, monkeypatch):
    setup = _setup(tmp_path, monkeypatch)
    calls = _network(monkeypatch)
    root, plan, _, registry, intended, limits, owner = setup
    with pytest.raises(FileExistsError):
        corpus.publish_corpus_plan(root=tmp_path / "duplicate", registry=registry, limits=limits,
                                  intended_issue_source=intended, owner=owner)
    with pytest.raises(transport.ResearchCaptureError, match="metadata changed"):
        corpus.run_corpus_batch(root=root, plan_sha256="f" * 64, identities=(),
            api_key="synthetic-test-only", capacity_snapshot=_capacity())
    assert not calls
    _run(setup)
    complete = transport.digest(transport.read_regular(root / "COMPLETE.json", corpus.MAX_METADATA))
    previous = corpus.CaptureReuse(root, plan, complete)
    next_root = tmp_path / "next"
    corpus.publish_corpus_plan(root=next_root, registry=registry, limits=limits,
        intended_issue_source=intended, owner=owner, previous=previous)
    assert corpus._json(next_root / "plan.json")["start_batch"] == 1
    with pytest.raises(FileExistsError):
        corpus.publish_corpus_plan(root=tmp_path / "second-next", registry=registry, limits=limits,
            intended_issue_source=intended, owner=owner, previous=previous)


def test_explicit_old_capture_reuse_requires_fresh_replay(tmp_path, monkeypatch):
    setup = _setup(tmp_path, monkeypatch)
    calls = _network(monkeypatch)
    first = _run(setup)
    source = setup[0] / "batches/batch-000000/capture"
    reuse = corpus.CaptureReuse(source, first["capture"]["plan_sha256"], first["capture"]["completion_sha256"])
    keys, proof = reuse.verify()
    assert len(keys) == 24 and proof["read_only_replay_verified"] and len(calls) == 24
    # Reuse does not accept a metadata-only hint or another completion identity.
    with pytest.raises(transport.ResearchCaptureError):
        replace(reuse, completion_sha256="0" * 64).verify()
    other = tmp_path / "reused-attempt"
    other.mkdir()
    reused_setup = _setup(other, monkeypatch)
    completed = _run(reused_setup, reuse=(reuse,))
    assert completed["capture"] is None and completed["actual"]["http_requests"] == 0
    assert _run(reused_setup)["range_complete"] and len(calls) == 24


def test_internally_valid_packed_store_must_bind_the_actual_capture(tmp_path, monkeypatch):
    setup = _setup(tmp_path, monkeypatch)
    calls = _network(monkeypatch)
    _run(setup)
    original = corpus.verify_packed_second_store

    def different_capture(**kwargs):
        result = original(**kwargs)
        result["source_provenance"]["capture_complete_sha256"] = "0" * 64
        return result

    monkeypatch.setattr(corpus, "verify_packed_second_store", different_capture)
    with pytest.raises(transport.ResearchCaptureError, match="different capture"):
        _run(setup)
    assert len(calls) == 24


def _replace_file(path, value):
    path.chmod(0o600)
    raw = transport.canonical(value)
    path.write_bytes(raw)
    return transport.digest(raw)


def _reseal_registry(registry, name, value, *, jsonl=False):
    path = registry.root / name
    raw = b"".join(transport.canonical(row) for row in value) if jsonl else transport.canonical(value)
    path.chmod(0o600)
    path.write_bytes(raw)
    complete = corpus._json(registry.root / "COMPLETE.json")
    for entry in complete["files"]:
        if entry["path"] == name:
            entry.update(bytes=len(raw), sha256=transport.digest(raw))
    return replace(registry, completion_sha256=_replace_file(registry.root / "COMPLETE.json", complete))


@pytest.mark.parametrize("change", [{"forward_fill_market_inputs": True}, {"covariates": True},
    {"news": True}, {"provider": "other"}, {"source": "minute_bars"},
    {"multiplier": True}, {"feature_engineering": 0}, {"availability_delay_ms": 900001},
    {"availability_assumption": "historical-finalized-assumed-delay"}])
def test_rehashed_registry_still_rejects_forbidden_raw_contract(tmp_path, change):
    registry, _ = _registry(tmp_path)
    summary = corpus._json(registry.root / "summary.json")
    summary["market_contract"].update(change)
    registry = _reseal_registry(registry, "summary.json", summary)
    with pytest.raises((ValueError, transport.ResearchCaptureError)):
        next(registry.batches())


@pytest.mark.parametrize("field,value", [("open_ms", float(START)), ("open_ms", START + 1),
    ("close_ms", START + 23400001), ("open_ms", True), ("partition_start", float(START))])
def test_rehashed_registry_requires_integral_second_aligned_clocks(tmp_path, field, value):
    registry, _ = _registry(tmp_path)
    rows = [transport.parse_json(raw) for raw in (registry.root / "sessions.jsonl").read_bytes().splitlines()]
    if field == "partition_start":
        rows[0]["partitions"][0]["start_ms"] = value
    else:
        rows[0][field] = value
    registry = _reseal_registry(registry, "sessions.jsonl", rows, jsonl=True)
    with pytest.raises(transport.ResearchCaptureError, match="Planning"):
        next(registry.batches())


@pytest.mark.parametrize("change", [{"start_batch": 1}, {"batch_count": 2},
    {"consumed_before": {"http_requests": 1, "raw_response_bytes": 0, "retained_bytes": 0}}])
def test_rehashed_plan_cannot_replace_immutable_owner_reservation(tmp_path, monkeypatch, change):
    setup = _setup(tmp_path, monkeypatch)
    calls = _network(monkeypatch)
    root, _, identities, *_ = setup
    plan = corpus._json(root / "plan.json")
    plan.update(change)
    new_sha = _replace_file(root / "plan.json", plan)
    with pytest.raises(transport.ResearchCaptureError, match="owner publication reservation"):
        corpus.run_corpus_batch(root=root, plan_sha256=new_sha, identities=identities,
            api_key="synthetic-test-only", capacity_snapshot=_capacity())
    assert not calls and not list((root / "batches").iterdir())


def test_owner_reservation_root_mismatch_is_rejected(tmp_path, monkeypatch):
    setup = _setup(tmp_path, monkeypatch)
    calls = _network(monkeypatch)
    owner = setup[-1]
    binding = corpus._json(owner / "initial.json")
    binding["root"] = str(tmp_path / "elsewhere")
    _replace_file(owner / "initial.json", binding)
    with pytest.raises(transport.ResearchCaptureError, match="owner publication reservation"):
        _run(setup)
    assert not calls


@pytest.mark.parametrize("change", [{"start_batch": 0}, {"batch_count": 2},
    {"consumed_before": {"http_requests": 0, "raw_response_bytes": 0, "retained_bytes": 0}}])
def test_even_rewritten_successor_reservation_cannot_erase_predecessor(tmp_path, monkeypatch, change):
    setup = _setup(tmp_path, monkeypatch)
    calls = _network(monkeypatch)
    _run(setup)
    root, prior_sha, _, registry, intended, limits, owner = setup
    complete_sha = transport.digest(transport.read_regular(root / "COMPLETE.json", corpus.MAX_METADATA))
    next_root = tmp_path / "next"
    corpus.publish_corpus_plan(root=next_root, registry=registry, limits=limits,
        intended_issue_source=intended, owner=owner, previous=corpus.CaptureReuse(root, prior_sha, complete_sha))
    plan = corpus._json(next_root / "plan.json")
    plan.update(change)
    new_sha = _replace_file(next_root / "plan.json", plan)
    # Even an operator-created replacement reservation cannot rewrite the
    # original externally hash-bound predecessor's completed range/counters.
    _replace_file(owner / (complete_sha + ".json"), {"root": str(next_root), "plan_sha256": new_sha})
    with pytest.raises(transport.ResearchCaptureError, match="range or predecessor counters"):
        corpus.run_corpus_batch(root=next_root, plan_sha256=new_sha, identities=(),
            api_key="synthetic-test-only", capacity_snapshot=_capacity())
    assert len(calls) == 24 and not list((next_root / "batches").iterdir())


def test_direct_storage_is_explicit_compact_and_replayed_before_reuse(tmp_path, monkeypatch):
    setup = _setup(tmp_path, monkeypatch, storage_mode=corpus.DIRECT_STORAGE)
    calls = _network(monkeypatch)
    plan = corpus._json(setup[0] / "plan.json")
    assert plan["schema"] == corpus.DIRECT_SCHEMA + "-plan"
    assert plan["storage_mode"] == corpus.DIRECT_STORAGE
    assert "raw_second_direct_capture_v1.py" in plan["implementation_sha256"]
    result = _run(setup)
    target = setup[0] / "batches/batch-000000"
    source = target / "capture"
    assert result["schema"] == corpus.DIRECT_SCHEMA + "-batch-complete"
    assert result["actual"]["http_requests"] == len(calls) == 24
    assert {p.name for p in source.iterdir()} == {"plan.json", "STARTED.json", "packed", "COMPLETE.json"}
    assert {p.name for p in (source / "packed").iterdir()} == {"index.json", "pages.gzpack"}
    assert not (target / "packed").exists()
    assert not list(source.glob("second-*"))
    started = corpus._json(target / "STARTED.json")
    assert started["reservation"]["retained_bytes"] == corpus.DIRECT_RETAINED_RESERVATION
    assert started["reservation"]["raw_response_bytes"] == corpus.capture.MAX_BYTES
    assert _run(setup)["range_complete"] and len(calls) == 24
    reuse = corpus.CaptureReuse(source, result["capture"]["plan_sha256"],
        result["capture"]["completion_sha256"], corpus.DIRECT_STORAGE)
    keys, proof = reuse.verify()
    assert len(keys) == proof["query_count"] == 24
    assert proof["packed_index_sha256"] == result["packed_index_sha256"]
    with pytest.raises(ValueError):
        replace(reuse, storage_mode=corpus.LEGACY_STORAGE).verify()
    other = tmp_path / "reuse"
    other.mkdir()
    reused_setup = _setup(other, monkeypatch, storage_mode=corpus.DIRECT_STORAGE)
    reused = _run(reused_setup, reuse=(reuse,))
    assert reused["capture"] is None and reused["actual"]["http_requests"] == 0
    assert _run(reused_setup)["range_complete"] and len(calls) == 24
    # Changing an original compressed frame invalidates both direct capture
    # reuse and the corpus prefix, without a new provider request.
    payload = source / "packed/pages.gzpack"
    raw = payload.read_bytes()
    payload.chmod(0o600)
    payload.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError):
        _run(setup)
    assert len(calls) == 24


def test_direct_tranche_cannot_switch_storage_generation_in_successor(tmp_path, monkeypatch):
    setup = _setup(tmp_path, monkeypatch, storage_mode=corpus.DIRECT_STORAGE)
    calls = _network(monkeypatch)
    _run(setup)
    root, plan_sha, _, registry, intended, limits, owner = setup
    complete_sha = transport.digest(transport.read_regular(root / "COMPLETE.json", corpus.MAX_METADATA))
    predecessor = corpus.CaptureReuse(root, plan_sha, complete_sha)
    with pytest.raises(transport.ResearchCaptureError, match="matching tranche"):
        corpus.publish_corpus_plan(root=tmp_path / "wrong", registry=registry, limits=limits,
            intended_issue_source=intended, owner=owner, previous=predecessor)
    assert not (tmp_path / "wrong").exists()
    successor = tmp_path / "next-direct"
    corpus.publish_corpus_plan(root=successor, registry=registry, limits=limits,
        intended_issue_source=intended, owner=owner, previous=predecessor,
        storage_mode=corpus.DIRECT_STORAGE)
    assert corpus._json(successor / "plan.json")["start_batch"] == 1 and len(calls) == 24


def test_direct_failed_batch_retains_original_and_is_never_retried(tmp_path, monkeypatch):
    setup = _setup(tmp_path, monkeypatch, storage_mode=corpus.DIRECT_STORAGE)
    calls = _network(monkeypatch, fail=True)
    with pytest.raises(transport.ResearchCaptureError, match="Capture incomplete"):
        _run(setup)
    source = setup[0] / "batches/batch-000000/capture"
    assert (source / "BLOCKED.json").is_file()
    assert (source / "packed/pages.gzpack").stat().st_size > 8
    assert not (source / "packed/index.json").exists()
    assert not (source / "COMPLETE.json").exists()
    with pytest.raises(transport.ResearchCaptureError, match="ambiguous"):
        _run(setup)
    assert len(calls) == 1


@pytest.mark.parametrize("storage_mode", ["packed", None, "legacy", "direct-packed-v2"])
def test_unknown_storage_mode_rejected_before_owner_publication(tmp_path, monkeypatch, storage_mode):
    with pytest.raises(transport.ResearchCaptureError, match="Unsupported explicit corpus storage"):
        _setup(tmp_path, monkeypatch, storage_mode=storage_mode)
    assert not (tmp_path / "owner").exists() and not (tmp_path / "attempt").exists()


def test_direct_raw_budget_is_reserved_before_any_request(tmp_path, monkeypatch):
    setup = _setup(tmp_path, monkeypatch, storage_mode=corpus.DIRECT_STORAGE,
                   maximum_raw_response_bytes=corpus.capture.MAX_BYTES - 1)
    calls = _network(monkeypatch)
    with pytest.raises(transport.ResearchCaptureError, match="Global corpus bound"):
        _run(setup)
    assert not calls and not list((setup[0] / "batches").iterdir())
