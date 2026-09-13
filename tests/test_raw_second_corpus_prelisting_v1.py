"""GPU-routed corpus accounting tests; synthetic sources and HTTP only.

The disposition validator is replaced for synthetic 24-slot accounting cases.
Original-source/issue/cutoff qualification belongs to the prelisting test suite.
The real typed serialization, corpus, capture, packing and replay remain active.
"""

from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path

import pytest
import torch

from rl_quant.data_sources.massive import raw_second_corpus_v1 as corpus
from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.raw_second_prelisting_v1 import SecondPrelistingDisposition
from test_raw_second_corpus_v1 import _capacity, _identity, _network, _registry, _replace_file, _write
from test_raw_second_prelisting_v1 import _disposition


pytestmark = pytest.mark.lsf_gpu


@pytest.fixture(scope="module", autouse=True)
def assigned_gpu():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1


def test_real_aapl_and_pltr_evidence_compose_without_a_registry_completion_claim(tmp_path, monkeypatch):
    """A diagnostic descriptor, not a claim that these are one registry batch."""
    originals = Path(os.environ["QT200_PRELISTING_EVIDENCE_ROOT"])
    aapl = Path(os.environ["QT200_ACTUAL_RELOCATED_ROOT"]) / "evidence/identity/response-0000.json"
    aapl_sha = "114962d1961d37c3872791d9c3a15da31e3e32198d68c757f6d789e3ac7b0704"
    assert sha256(transport.read_regular(aapl, 65536)).hexdigest() == aapl_sha
    disposition = _disposition(originals, "PLTR")
    ordinary = corpus.CorpusIdentity("AAPL", "AAPL",
        corpus.AliasIdentityRef("dated-ticker-wrapper", str(aapl), aapl_sha))
    source_paths = {aapl, Path(disposition.identity.path), Path(disposition.intended_issue_source)}
    source_paths.update(Path(path) for source in disposition.primary_sources
                        for path in (source.path, source.receipt_path))
    before = {str(path): sha256(transport.read_regular(path, 8 * 1024**2)).hexdigest() for path in source_paths}

    def denied(*args, **kwargs):
        raise AssertionError("Real-evidence composition cannot issue provider requests")

    monkeypatch.setattr(transport, "_fetch", denied)
    monkeypatch.setattr(transport.request, "build_opener", denied)
    assert SecondPrelistingDisposition.validate is not _synthetic_disposition
    batch = {"diagnostic_only": True, "registry_batch_claimed": False,
        "session_date": "2017-01-03", "start_ms": 1483453800000, "end_ms": 1483457399000,
        "fixed_slot_tickers": ["AAPL", "PLTR"],
        "diagnostic_original_slot_indices": [corpus.SYMBOLS.index(name) for name in ("AAPL", "PLTR")]}
    plan = {"slot_accounting": corpus.SLOT_ACCOUNTING,
        "intended_issue_source": disposition.intended_issue_source,
        "intended_issue_sha256": disposition.intended_issue_sha256}
    queries, aliases, identity_proofs, slots = corpus._slot_queries(plan, batch, (ordinary,), (disposition,))
    assert [asdict(query) for query in queries] == [
        {"ticker": "AAPL", "start_ms": batch["start_ms"], "end_ms": batch["end_ms"]}]
    assert not aliases and len(identity_proofs) == 1
    assert slots[0]["fixed_slot_ticker"] == corpus.SYMBOLS[0] == "AAPL"
    assert [row["disposition"] for row in slots] == ["acquisition-required", "known-unavailable"]
    assert slots[1]["proof"]["fixed_slot_index"] == corpus.SYMBOLS.index("PLTR")
    assert slots[1]["proof"]["fixed_slot_index"] == batch["diagnostic_original_slot_indices"][1] > 1
    assert slots[1]["proof"]["provider_capture_reference"] is None
    assert slots[1]["proof"]["point_in_time_qualified"] is False
    assert slots[1]["proof"]["training_ready"] is False
    restored = SecondPrelistingDisposition.from_dict(json.loads(json.dumps(disposition.to_dict())))
    assert corpus._slot_queries(plan, batch, (ordinary,), (restored,)) == (queries, aliases, identity_proofs, slots)
    assert {str(path): sha256(transport.read_regular(path, 8 * 1024**2)).hexdigest()
            for path in source_paths} == before
    assert not list(tmp_path.iterdir())  # No owner, plan, capture, pack or COMPLETE is emitted.


def _synthetic_disposition(self, query):
    source = transport.read_regular(Path(self.identity.path), 65536)
    corpus._require(transport.digest(source) == self.identity.sha256, "Synthetic evidence changed")
    corpus._require(query.ticker == self.fixed_slot_ticker
                    and self.session_date < self.public_trading_start_date,
                    "Synthetic cutoff or slot differs")
    return {"schema": "synthetic-prelisting-accounting-proof-v1", "query": asdict(query),
        "fixed_slot_ticker": self.fixed_slot_ticker, "session_date": self.session_date,
        "source_sha256": self.identity.sha256, "acquisition_required": False,
        "market_observation_status": "known-unavailable",
        "reason": "intended-class-before-public-trading", "training_ready": False}


def _setup(tmp_path, monkeypatch, *, mode=corpus.DIRECT_STORAGE, enabled=True, tranche=1,
           unavailable_count=1, limits=None):
    monkeypatch.setattr(corpus, "validate_fixed_slot_identity", _identity)
    monkeypatch.setattr(SecondPrelistingDisposition, "validate", _synthetic_disposition)
    registry, intended = _registry(tmp_path)
    evidence = tmp_path / "synthetic-disposition-source.json"
    evidence_sha = _write(evidence, {"synthetic_accounting_only": True})
    maximums = {"maximum_http_requests": 10000, "maximum_raw_response_bytes": 1000000000,
        "maximum_retained_bytes": 1000000000, "project_ceiling_bytes": 900000000000,
        "tranche_batches": tranche, **(limits or {})}
    bound = corpus.CorpusLimits(**maximums)
    root, owner = tmp_path / "attempt", tmp_path / "owner"
    plan_sha = corpus.publish_corpus_plan(root=root, registry=registry, limits=bound,
        intended_issue_source=intended, owner=owner, storage_mode=mode,
        slot_accounting=corpus.SLOT_ACCOUNTING if enabled else None)
    plan = corpus._json(root / "plan.json")

    def routes(ordinal=0, count=unavailable_count):
        batch = next(registry.batches(ordinal))
        names = batch["fixed_slot_tickers"]
        ordinary = tuple(corpus.CorpusIdentity(name, name,
            corpus.AliasIdentityRef("dated-ticker-wrapper", str(intended), "a" * 64))
            for name in names[:len(names) - count])
        absent = tuple(SecondPrelistingDisposition(fixed_slot_ticker=name,
            session_date=batch["session_date"], public_trading_start_date="2020-01-01",
            identity=corpus.AliasIdentityRef("dated-ticker-wrapper", str(evidence), evidence_sha),
            identity_observation_date="2022-01-03", intended_issue_source=str(intended),
            intended_issue_sha256=plan["intended_issue_sha256"], primary_sources=())
            for name in names[len(names) - count:])
        return ordinary, absent

    def run(*, ordinal=0, count=unavailable_count, capacity=None, **kwargs):
        ordinary, absent = routes(ordinal, count)
        return corpus.run_corpus_batch(root=root, plan_sha256=plan_sha,
            identities=ordinary, unavailable=absent, api_key="synthetic-test-only",
            capacity_snapshot=_capacity() if capacity is None else capacity, **kwargs)

    return root, plan_sha, run, routes, registry, intended, bound, owner, evidence


@pytest.mark.parametrize("mode", [corpus.LEGACY_STORAGE, corpus.DIRECT_STORAGE])
def test_mixed_fresh_reused_and_unavailable_preserve_all_original_slots(tmp_path, monkeypatch, mode):
    setup = _setup(tmp_path, monkeypatch, mode=mode, unavailable_count=2)
    root, _, run, routes, *_ = setup
    calls = _network(monkeypatch)
    ordinary, _ = routes()
    publish, acquire, _ = corpus._capture_api(mode)
    source = tmp_path / "reusable-one-query"
    query = corpus.SecondQuery(ordinary[0].provider_ticker, 1483453800000, 1483457399000)
    source_sha = publish(root=source, queries=(query,))
    acquire(root=source, plan_sha256=source_sha, api_key="synthetic-test-only")
    completion_sha = transport.digest(transport.read_regular(source / "COMPLETE.json", corpus.MAX_METADATA))
    ref = corpus.CaptureReuse(source, source_sha, completion_sha, mode)
    terminal = run(reuse=(ref,))
    accounting = terminal["slot_accounting"]
    assert accounting["slot_count"] == 24 and accounting["fixed_universe_size"] == 200
    assert accounting["acquired_queries"] == 21 and accounting["reused_queries"] == 1
    assert accounting["known_unavailable_slots"] == 2
    assert [r["fixed_slot_ticker"] for r in accounting["slots"]] == list(corpus.SYMBOLS[:24])
    assert [r["fixed_slot_index"] for r in accounting["slots"]] == list(range(24))
    assert all(row["query"] is None for row in accounting["slots"][-2:])
    assert terminal["actual"]["http_requests"] == 21 and len(calls) == 22
    captured = corpus._json(root / "batches/batch-000000/capture/plan.json")["queries"]
    assert [q["ticker"] for q in captured] == list(corpus.SYMBOLS[1:22])
    complete = run()
    assert complete["slot_totals"] == {"slot_count": 24, "acquired_queries": 21,
        "reused_queries": 1, "known_unavailable_slots": 2}
    assert complete["range_complete"] and len(calls) == 22


@pytest.mark.parametrize("mode", [corpus.LEGACY_STORAGE, corpus.DIRECT_STORAGE])
def test_all_unavailable_has_zero_http_no_fake_capture_and_replays(tmp_path, monkeypatch, mode):
    root, _, run, _, _, _, _, _, _ = _setup(tmp_path, monkeypatch, mode=mode,
        unavailable_count=24, limits={"maximum_http_requests": 1, "maximum_raw_response_bytes": 1})
    calls = _network(monkeypatch)
    monkeypatch.setattr(corpus, "_capture_api", lambda *_: pytest.fail("No capture API for prelisting-only batch"))
    terminal = run()
    target = root / "batches/batch-000000"
    assert not calls and set(p.name for p in target.iterdir()) == {"STARTED.json", "COMPLETE.json"}
    assert terminal["capture"] is None and terminal["packed_index_sha256"] is None
    assert terminal["actual"]["http_requests"] == terminal["actual"]["raw_response_bytes"] == 0
    assert terminal["actual"]["retained_bytes"] > 0
    assert terminal["slot_accounting"]["known_unavailable_slots"] == 24
    started = corpus._json(target / "STARTED.json")
    assert started["reservation"]["http_requests"] == started["reservation"]["raw_response_bytes"] == 0
    complete = run()
    assert complete["next_batch"] == 1 and complete["slot_totals"]["slot_count"] == 24
    assert not complete["training_ready"] and not complete["training_authorized"]


def test_reused_plus_unavailable_has_no_fresh_capture(tmp_path, monkeypatch):
    root, _, run, routes, *_ = _setup(tmp_path, monkeypatch, unavailable_count=2)
    calls = _network(monkeypatch)
    ordinary, _ = routes()
    queries = tuple(corpus.SecondQuery(i.provider_ticker, 1483453800000, 1483457399000) for i in ordinary)
    publish, acquire, _ = corpus._capture_api(corpus.DIRECT_STORAGE)
    source = tmp_path / "reused-query-population"
    source_sha = publish(root=source, queries=queries)
    acquire(root=source, plan_sha256=source_sha, api_key="synthetic-test-only")
    complete_sha = transport.digest(transport.read_regular(source / "COMPLETE.json", corpus.MAX_METADATA))
    terminal = run(reuse=(corpus.CaptureReuse(source, source_sha, complete_sha, corpus.DIRECT_STORAGE),))
    assert len(calls) == 22 and terminal["actual"]["http_requests"] == 0
    assert terminal["actual"]["raw_response_bytes"] == 0 and terminal["capture"] is None
    assert terminal["slot_accounting"]["acquired_queries"] == 0
    assert terminal["slot_accounting"]["reused_queries"] == 22
    assert terminal["slot_accounting"]["known_unavailable_slots"] == 2
    assert not (root / "batches/batch-000000/capture").exists()
    assert run()["range_complete"] and len(calls) == 22


def test_one_hour_accounts_for_all_200_slots_including_final_eight(tmp_path, monkeypatch):
    _, _, run, *_ = _setup(tmp_path, monkeypatch, unavailable_count=24, tranche=9)
    calls = _network(monkeypatch)
    positions = []
    for ordinal in range(9):
        terminal = run(ordinal=ordinal, count=8 if ordinal == 8 else 24)
        positions.extend(row["fixed_slot_index"] for row in terminal["slot_accounting"]["slots"])
    assert positions == list(range(200)) and not calls
    complete = run()
    assert complete["next_batch"] == 9
    assert complete["slot_totals"] == {"slot_count": 200, "acquired_queries": 0,
        "reused_queries": 0, "known_unavailable_slots": 200}


@pytest.mark.parametrize("change", ["missing", "duplicate", "overlap", "reordered", "untyped"])
def test_invalid_slot_cover_rejected_before_any_batch_write(tmp_path, monkeypatch, change):
    root, plan_sha, _, routes, *_ = _setup(tmp_path, monkeypatch, unavailable_count=2)
    calls = _network(monkeypatch)
    ordinary, absent = routes()
    if change == "missing":
        absent = absent[:-1]
    elif change == "duplicate":
        absent = absent + absent[:1]
    elif change == "overlap":
        absent = (replace(absent[0], fixed_slot_ticker=ordinary[0].fixed_slot_ticker), absent[1])
    elif change == "reordered":
        absent = absent[::-1]
    else:
        absent = tuple(d.to_dict() for d in absent)
    with pytest.raises(transport.ResearchCaptureError, match="slot disposition"):
        corpus.run_corpus_batch(root=root, plan_sha256=plan_sha, identities=ordinary,
            unavailable=absent, api_key="synthetic-test-only", capacity_snapshot=_capacity())
    assert not calls and not list((root / "batches").iterdir())


@pytest.mark.parametrize("change", ["date", "intended", "cutoff"])
def test_disposition_binding_rejected_before_any_batch_write(tmp_path, monkeypatch, change):
    root, plan_sha, _, routes, *_ = _setup(tmp_path, monkeypatch)
    calls = _network(monkeypatch)
    ordinary, absent = routes()
    updates = {"date": {"session_date": "2017-01-04"},
        "intended": {"intended_issue_sha256": "f" * 64},
        "cutoff": {"public_trading_start_date": "2017-01-03"}}[change]
    with pytest.raises(transport.ResearchCaptureError):
        corpus.run_corpus_batch(root=root, plan_sha256=plan_sha, identities=ordinary,
            unavailable=(replace(absent[0], **updates),), api_key="synthetic-test-only", capacity_snapshot=_capacity())
    assert not calls and not list((root / "batches").iterdir())


@pytest.mark.parametrize("failure", ["capacity", "metadata_budget"])
def test_no_http_batch_still_requires_capacity_and_retained_budget(tmp_path, monkeypatch, failure):
    limits = {"maximum_retained_bytes": 1024} if failure == "metadata_budget" else None
    root, _, run, *_ = _setup(tmp_path, monkeypatch, unavailable_count=24, limits=limits)
    calls = _network(monkeypatch)
    with pytest.raises(transport.ResearchCaptureError):
        run(capacity=_capacity(fileset_available_bytes=0) if failure == "capacity" else None)
    assert not calls and not list((root / "batches").iterdir())


@pytest.mark.parametrize("change", ["source", "count", "disposition", "invented_capture", "tranche_count"])
def test_completed_prelisting_replay_revalidates_evidence_and_accounting(tmp_path, monkeypatch, change):
    root, _, run, _, _, _, _, _, evidence = _setup(tmp_path, monkeypatch, unavailable_count=24)
    calls = _network(monkeypatch)
    run()
    target = root / "batches/batch-000000"
    if change == "source":
        _replace_file(evidence, {"changed": True})
    elif change == "invented_capture":
        (target / "capture").mkdir()
    elif change == "tranche_count":
        path = root / "COMPLETE.json"
        terminal = corpus._json(path)
        terminal["slot_totals"]["known_unavailable_slots"] = 0
        _replace_file(path, terminal)
    else:
        path = target / "COMPLETE.json"
        terminal = corpus._json(path)
        if change == "count":
            terminal["slot_accounting"]["known_unavailable_slots"] = 0
        else:
            terminal["slot_accounting"]["slots"][0]["disposition"] = "reused"
        _replace_file(path, terminal)
    with pytest.raises(transport.ResearchCaptureError):
        run()
    assert not calls


def test_unknown_next_identity_does_not_skip_ordinal_or_complete_subset(tmp_path, monkeypatch):
    root, plan_sha, run, routes, *_ = _setup(tmp_path, monkeypatch, unavailable_count=24, tranche=2)
    calls = _network(monkeypatch)
    run()
    ordinary, absent = routes(1, 24)
    with pytest.raises(transport.ResearchCaptureError, match="slot disposition"):
        corpus.run_corpus_batch(root=root, plan_sha256=plan_sha, identities=ordinary,
            unavailable=absent[:-1], api_key="synthetic-test-only", capacity_snapshot=_capacity())
    assert not (root / "COMPLETE.json").exists()
    assert {p.name for p in (root / "batches").iterdir()} == {"batch-000000"}
    assert not calls
    run(ordinal=1)
    complete = run()
    assert complete["next_batch"] == 2 and complete["slot_totals"]["known_unavailable_slots"] == 48


def test_legacy_generation_rejects_dispositions_and_keeps_old_output(tmp_path, monkeypatch):
    root, _, run, *_ = _setup(tmp_path, monkeypatch, enabled=False)
    calls = _network(monkeypatch)
    with pytest.raises(transport.ResearchCaptureError, match="explicit slot accounting"):
        run()
    assert not calls and not list((root / "batches").iterdir())
    terminal = run(count=0)
    assert "slot_accounting" not in terminal
    assert "slot_accounting" not in corpus._json(root / "plan.json")
    assert "raw_second_prelisting_v1.py" not in corpus._json(root / "plan.json")["implementation_sha256"]
    assert "slot_totals" not in run()


def test_new_generation_pins_prelisting_source_and_rejects_old_schema_switch(tmp_path, monkeypatch):
    root, _, _, _, _, _, _, _, _ = _setup(tmp_path, monkeypatch)
    plan = corpus._json(root / "plan.json")
    assert "raw_second_prelisting_v1.py" in plan["implementation_sha256"]
    assert plan["schema"] == corpus.DIRECT_SCHEMA + "-slots-v1-plan"
    plan["schema"] = corpus.DIRECT_SCHEMA + "-plan"
    with pytest.raises(transport.ResearchCaptureError, match="generation"):
        corpus._storage(plan)


def test_successive_new_tranches_preserve_counters_and_disallow_generation_switch(tmp_path, monkeypatch):
    root, plan_sha, run, routes, registry, intended, limits, owner, _ = _setup(
        tmp_path, monkeypatch, unavailable_count=24)
    calls = _network(monkeypatch)
    run()
    complete = run()
    terminal_sha = transport.digest(transport.read_regular(root / "COMPLETE.json", corpus.MAX_METADATA))
    previous = corpus.CaptureReuse(root, plan_sha, terminal_sha, corpus.DIRECT_STORAGE)
    with pytest.raises(transport.ResearchCaptureError, match="matching tranche"):
        corpus.publish_corpus_plan(root=tmp_path / "wrong-generation", registry=registry,
            limits=limits, intended_issue_source=intended, owner=owner, previous=previous,
            storage_mode=corpus.DIRECT_STORAGE)
    next_root = tmp_path / "next-tranche"
    next_sha = corpus.publish_corpus_plan(root=next_root, registry=registry, limits=limits,
        intended_issue_source=intended, owner=owner, previous=previous, storage_mode=corpus.DIRECT_STORAGE,
        slot_accounting=corpus.SLOT_ACCOUNTING)
    next_plan = corpus._json(next_root / "plan.json")
    assert next_plan["start_batch"] == 1 and next_plan["consumed_before"] == complete["consumed"]
    ordinary, absent = routes(1, 24)
    terminal = corpus.run_corpus_batch(root=next_root, plan_sha256=next_sha, identities=ordinary,
        unavailable=absent, api_key="", capacity_snapshot=_capacity())
    assert terminal["batch"]["ordinal"] == 1 and not calls
    assert terminal["consumed_after"]["retained_bytes"] > complete["consumed"]["retained_bytes"]
    with pytest.raises(FileExistsError):
        corpus.publish_corpus_plan(root=tmp_path / "duplicate-successor", registry=registry,
            limits=limits, intended_issue_source=intended, owner=owner, previous=previous,
            storage_mode=corpus.DIRECT_STORAGE, slot_accounting=corpus.SLOT_ACCOUNTING)
