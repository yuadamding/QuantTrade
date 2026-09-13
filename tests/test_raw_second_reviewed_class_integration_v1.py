"""Persisted corpus tests for the explicit HLT/RCL/CCL acquisition routes.

These use original class evidence with mocked bars, not an actual-data claim.
Unrelated ordinary slots use the existing synthetic identity fixture; the
three reviewed routes and their original-byte checks are never monkeypatched.
"""

from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path

import pytest
import torch

from rl_quant.data_sources.massive import raw_second_corpus_v1 as corpus
from rl_quant.data_sources.massive import raw_second_reviewed_class_v1 as classes
from rl_quant.data_sources.massive import qt200_research_capture_v1 as io
from rl_quant.data_sources.massive import raw_second_direct_capture_v1 as direct
from test_raw_second_corpus_v1 import _registry, _reseal_registry, _identity, _network, _capacity
from test_raw_second_reviewed_class_v1 import (
    originals as originals, _route, _copy, _query, _inventory, _denied, _TICKERS,
)

pytestmark = pytest.mark.lsf_gpu


@pytest.fixture(autouse=True, scope="module")
def assigned_gpu():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1


def _fixture(tmp_path, originals, monkeypatch, *, enabled=True):
    routes = {t: _copy(tmp_path / "evidence" / t, originals, t) for t in _TICKERS}
    # Each route must bind the SAME immutable intended-population path.
    intended = routes["HLT"].intended_issue_source
    routes = {t: replace(r, intended_issue_source=intended) for t, r in routes.items()}
    registry, _ = _registry(tmp_path)
    summary = corpus._json(registry.root / "summary.json")
    summary["identity_evidence"]["sha256"] = routes["HLT"].intended_issue_sha256
    registry = _reseal_registry(registry, "summary.json", summary)
    monkeypatch.setattr(corpus, "validate_fixed_slot_identity", _identity)
    limits = corpus.CorpusLimits(2000, 768_000_000, 800_000_000, 700_000_000_000, tranche_batches=3)
    root, owner = tmp_path / "corpus", tmp_path / "owner"
    plan = corpus.publish_corpus_plan(root=root, registry=registry, limits=limits,
        intended_issue_source=Path(intended), owner=owner, storage_mode=corpus.DIRECT_STORAGE,
        slot_accounting=corpus.SLOT_ACCOUNTING,
        class_routing=corpus.REVIEWED_CLASS_ROUTING if enabled else None)

    def identities(ordinal):
        batch = next(registry.batches(ordinal))
        return tuple(corpus.CorpusIdentity(name, name, routes[name].identity, reviewed_class=routes[name])
                     if name in routes else corpus.CorpusIdentity(name, name, routes["HLT"].modern_identity)
                     for name in batch["fixed_slot_tickers"])

    def run(ordinal, **kwargs):
        return corpus.run_corpus_batch(root=root, plan_sha256=plan, identities=identities(ordinal),
            api_key="synthetic-test-only", capacity_snapshot=_capacity(project_ceiling_bytes=700_000_000_000), **kwargs)

    return root, plan, registry, limits, owner, routes, identities, run


def test_legacy_fields_stay_exact_and_new_type_has_explicit_discriminator(originals):
    route = _route(originals, "HLT")
    old = corpus.CorpusIdentity("HLT", "HLT", route.identity)
    assert old.to_dict() == dict(fixed_slot_ticker="HLT", provider_ticker="HLT", identity=asdict(route.identity),
                                 alias=None, successor=None)
    identity = replace(old, reviewed_class=route)
    assert identity.to_dict() == {**old.to_dict(), "reviewed_class": route.to_dict()}
    module = "raw_second_reviewed_class_v1.py"
    assert module not in corpus._implementation(corpus.DIRECT_STORAGE)
    assert corpus._implementation(corpus.DIRECT_STORAGE, class_routing=corpus.REVIEWED_CLASS_ROUTING)[module] == (
        sha256(Path(classes.__file__).read_bytes()).hexdigest())


@pytest.mark.parametrize("ticker", _TICKERS)
@pytest.mark.parametrize("case", ["no-opt-in", "wrong-opt-in", "intended-path", "intended-hash", "slot",
                                   "outer-identity", "date", "mixed-successor", "mixed-alias", "untyped"])
def test_exact_binding_rejects_before_writes_or_requests(originals, tmp_path, monkeypatch, ticker, case):
    route = _route(originals, ticker)
    plan = dict(class_routing=corpus.REVIEWED_CLASS_ROUTING, intended_issue_source=route.intended_issue_source,
                intended_issue_sha256=route.intended_issue_sha256)
    query = _query(ticker)
    batch = dict(session_date=route.session_date, start_ms=query.start_ms, end_ms=query.end_ms,
                 fixed_slot_tickers=[ticker], diagnostic_only=True)
    identity = corpus.CorpusIdentity(ticker, ticker, route.identity, reviewed_class=route)
    if case == "no-opt-in":
        plan.pop("class_routing")
    elif case == "wrong-opt-in":
        plan["class_routing"] = "generic-nullable-figi"
    elif case == "intended-path":
        plan["intended_issue_source"] += ".different"
    elif case == "intended-hash":
        plan["intended_issue_sha256"] = "0" * 64
    elif case == "slot":
        identity = replace(identity, fixed_slot_ticker="AAPL")
    elif case == "outer-identity":
        identity = replace(identity, identity=route.modern_identity)
    elif case == "date":
        batch["session_date"] = "2017-01-04"
    elif case == "mixed-successor":
        identity = replace(identity, successor=object())
    elif case == "mixed-alias":
        identity = replace(identity, alias=object())
    else:
        identity = replace(identity, reviewed_class=route.to_dict())
    monkeypatch.setattr(io, "_fetch", _denied)
    monkeypatch.setattr(io.request, "build_opener", _denied)
    with pytest.raises(ValueError):
        corpus._identity_queries(plan, batch, (identity,))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("enabled", [False, True])
def test_all_three_real_routes_survive_full_prefix_capture_and_readonly_replay(originals, tmp_path, monkeypatch, enabled):
    root, plan_sha, registry, limits, owner, routes, identities, run = _fixture(
        tmp_path, originals, monkeypatch, enabled=enabled)
    calls = _network(monkeypatch)
    source_before = _inventory(tmp_path / "evidence")
    run(0)
    run(1)
    assert len(calls) == 48
    target = root / "batches/batch-000002"
    if not enabled:
        before = _inventory(root)
        with pytest.raises(ValueError, match="opt-in"):
            run(2)
        assert len(calls) == 48 and _inventory(root) == before and not target.exists()
        return
    result = run(2)
    assert len(calls) == 72 and result["actual"]["http_requests"] == 24
    started = corpus._json(target / "STARTED.json")
    assert started["batch"]["fixed_slot_tickers"] == list(corpus.SYMBOLS[48:72])
    for ticker, index in (("HLT", 11), ("RCL", 12), ("CCL", 13)):
        assert started["identity_routes"][index] == identities(2)[index].to_dict()
        assert started["identity_proofs"][index] == routes[ticker].validate(_query(ticker))
        assert result["slot_accounting"]["slots"][index]["fixed_slot_index"] == 48 + index
    native = direct.replay_direct_second_capture(root=target / "capture",
        plan_sha256=result["capture"]["plan_sha256"], completion_sha256=result["capture"]["completion_sha256"])
    assert native["query_count"] == 24 and native["training_ready"] is False
    before = _inventory(root)
    monkeypatch.setattr(io, "_fetch", _denied)
    monkeypatch.setattr(io.request, "build_opener", _denied)
    complete = run(2)
    assert complete["range_complete"] is True and complete["next_batch"] == 3
    assert complete["training_ready"] is False and complete["training_authorized"] is False
    assert _inventory(root) == before and _inventory(tmp_path / "evidence") == source_before
    ref = corpus.CaptureReuse(root, plan_sha, sha256((root / "COMPLETE.json").read_bytes()).hexdigest())
    # A successor cannot silently drop the exact class policy from its lineage.
    with pytest.raises(ValueError, match="Predecessor"):
        corpus.publish_corpus_plan(root=tmp_path / "bad-next", registry=registry, limits=limits,
            intended_issue_source=Path(routes["HLT"].intended_issue_source), owner=owner, previous=ref,
            storage_mode=corpus.DIRECT_STORAGE, slot_accounting=corpus.SLOT_ACCOUNTING)
    assert not (tmp_path / "bad-next").exists()
    next_sha = corpus.publish_corpus_plan(root=tmp_path / "next", registry=registry, limits=limits,
        intended_issue_source=Path(routes["HLT"].intended_issue_source), owner=owner, previous=ref,
        storage_mode=corpus.DIRECT_STORAGE, slot_accounting=corpus.SLOT_ACCOUNTING,
        class_routing=corpus.REVIEWED_CLASS_ROUTING)
    assert corpus._json(tmp_path / "next/plan.json", next_sha)["start_batch"] == 3
    assert len(calls) == 72


@pytest.mark.parametrize("ticker", _TICKERS)
def test_changed_primary_original_blocks_even_completed_prefix_replay(originals, tmp_path, monkeypatch, ticker):
    root, _, _, _, _, routes, _, run = _fixture(tmp_path, originals, monkeypatch)
    calls = _network(monkeypatch)
    for ordinal in range(3):
        run(ordinal)
    assert len(calls) == 72
    before = _inventory(root)
    Path(routes[ticker].primary_sources[0].path).write_bytes(b"changed original source")
    monkeypatch.setattr(io, "_fetch", _denied)
    with pytest.raises(ValueError):
        run(2)
    assert _inventory(root) == before and len(calls) == 72


@pytest.mark.parametrize("setting", [False, "", "same-cik", "reviewed-common-class-v2"])
def test_unknown_class_policy_rejects_before_plan_publication(tmp_path, monkeypatch, setting):
    registry, intended = _registry(tmp_path)
    with pytest.raises(ValueError):
        corpus.publish_corpus_plan(root=tmp_path / "corpus", registry=registry,
            limits=corpus.CorpusLimits(2000, 768_000_000, 800_000_000, 700_000_000_000, tranche_batches=3),
            intended_issue_source=intended, owner=tmp_path / "owner", class_routing=setting)
    assert not (tmp_path / "corpus").exists() and not (tmp_path / "owner").exists()
