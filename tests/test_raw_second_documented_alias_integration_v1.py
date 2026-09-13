"""LSF-only BKNG route integration; retained identities, synthetic HTTP only.

Reuse the nine original fixtures from test_raw_second_documented_alias_v1.
The three-batch corpus case preserves real registry coordinates (BKNG is 56),
but mocks every ordinary issue validator. Only its BKNG validator is genuine;
these tests do not qualify the other 71 historical issues or an actual corpus.
"""

from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive import raw_second_capture_v1 as capture
from rl_quant.data_sources.massive import raw_second_corpus_v1 as corpus
from rl_quant.data_sources.massive import raw_second_direct_capture_v1 as direct
from rl_quant.data_sources.massive import raw_second_documented_alias_v1 as documented
from rl_quant.datasets.massive_raw_second_packed_v1 import PackedSecondCaptureRef
from rl_quant.datasets.massive_raw_seconds_v1 import SecondCaptureRef, SecondQuery
from test_raw_second_alias_capture_v1 import _route as _legacy_route
from test_raw_second_corpus_v1 import (
    _capacity, _identity, _network as _corpus_network, _registry, _replace_file, _reseal_registry,
)
from test_raw_second_direct_capture_v1 import KEY, _body, _network, _url
from test_raw_second_documented_alias_v1 import (
    _copied_route, _denied, _inventory, _query, _route, originals as originals,
)
from test_raw_second_evidence_v1 import _seal

pytestmark = pytest.mark.lsf_gpu
_OPTIONAL = {"documented_alias_schema", "documented_alias_implementation_sha256"}
_LOOSE_KEYS = {
    "schema", "queries", "maximum_raw_response_bytes", "maximum_page_bytes", "maximum_pages_per_query",
    "maximum_elapsed_seconds", "retry_count", "concurrent_requests", "minimum_request_gap_seconds",
    "market_fields", "training_ready", "point_in_time_qualified", "native_v5_qualified",
    "capture_implementation_sha256", "second_implementation_sha256",
}
_DIRECT_KEYS = {
    "schema", "queries", "alias_routes", "maximum_raw_response_bytes", "maximum_page_bytes",
    "maximum_pages_per_query", "maximum_elapsed_seconds", "maximum_queries", "maximum_query_seconds",
    "maximum_packed_bytes", "maximum_index_bytes", "maximum_frame_header_bytes", "maximum_receipt_bytes",
    "maximum_request_url_bytes", "maximum_query_metadata_bytes", "maximum_census_bytes",
    "maximum_root_metadata_bytes", "storage_reserve_bytes", "allocation_reserve_bytes",
    "maximum_page_working_memory_bytes", "pre_request_reservation", "retry_count", "concurrent_requests",
    "minimum_request_gap_seconds", "market_fields", "continuous_identity_qualified",
    "provider_tickers_rewritten", "training_ready", "point_in_time_qualified", "native_v5_qualified",
    "capture_implementation_sha256", "second_implementation_sha256", "alias_implementation_sha256",
    "packed_implementation_sha256", "raw_loader_implementation_sha256", "evidence_implementation_sha256",
    "direct_implementation_sha256",
}


@pytest.fixture(scope="module", autouse=True)
def assigned_gpu():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1


def _api(mode):
    return ((capture.publish_second_capture_plan, capture.capture_seconds, capture.replay_second_capture)
            if mode == "loose" else (direct.publish_direct_second_capture_plan,
                direct.capture_direct_seconds, direct.replay_direct_second_capture))


@pytest.mark.parametrize("mode", ["loose", "direct"])
def test_documented_capture_persists_literal_pcln_and_replays_original_pages(originals, tmp_path, monkeypatch, mode):
    route, query = _route(originals), _query()
    publish, acquire, replay = _api(mode)
    source_before = _inventory(originals)
    root = tmp_path / "capture"
    plan_sha = publish(root=root, queries=(query,), alias_routes=(route,))
    plan = json.loads((root / "plan.json").read_bytes())
    assert plan["queries"] == [asdict(query)]
    assert plan["alias_routes"] == [route.to_dict()]
    assert plan["documented_alias_schema"] == documented.SCHEMA
    assert plan["documented_alias_implementation_sha256"] == sha256(
        Path(documented.__file__).read_bytes()).hexdigest()
    assert type(capture._alias_route_from_dict(plan["alias_routes"][0])) is documented.SecondDocumentedAliasRoute
    assert capture._alias_route_from_dict(plan["alias_routes"][0]) == route
    bodies = [_body(query, 0, next_page=1), _body(query, 1)]
    calls = _network(monkeypatch, [(_url(query, i), 200, body) for i, body in enumerate(bodies)])
    acquire(root=root, plan_sha256=plan_sha, api_key=KEY)
    complete_sha = sha256((root / "COMPLETE.json").read_bytes()).hexdigest()
    result = replay(root=root, plan_sha256=plan_sha, completion_sha256=complete_sha)
    if mode == "loose":
        materialized = capture.materialize_second_sources(root=root, plan_sha256=plan_sha,
            completion_sha256=complete_sha, output=tmp_path / "native")
        assert materialized["alias_routes"] == [route.to_dict()]
        assert materialized["documented_alias_schema"] == documented.SCHEMA
        ref = SecondCaptureRef(**materialized["captures"][0])
    else:
        ref = PackedSecondCaptureRef.from_dict(result["captures"][0])
    loaded_query, pages = ref.load()
    assert loaded_query == query and loaded_query.ticker == "PCLN"
    assert [page.body for page in pages] == bodies
    assert all(json.loads(page.body)["ticker"] == "PCLN" for page in pages)
    assert [page.request_url for page in pages] == [_url(query, 0), _url(query, 1)]
    assert result["page_count"] == len(calls) == 2 and result["training_ready"] is False
    before = _inventory(tmp_path)
    monkeypatch.setattr(transport, "_fetch", _denied)
    monkeypatch.setattr(transport.request, "build_opener", _denied)
    assert replay(root=root, plan_sha256=plan_sha, completion_sha256=complete_sha) == result
    assert _inventory(tmp_path) == before and _inventory(originals) == source_before


@pytest.mark.parametrize("mode", ["loose", "direct"])
def test_default_plan_keysets_and_legacy_alias_payloads_remain_unchanged(tmp_path, mode):
    publish, *_ = _api(mode)
    ordinary = tmp_path / "ordinary"
    publish(root=ordinary, queries=(SecondQuery("AAPL", 1000, 2000),))
    plan = json.loads((ordinary / "plan.json").read_bytes())
    assert set(plan) == (_LOOSE_KEYS if mode == "loose" else _DIRECT_KEYS)
    assert not _OPTIONAL & set(plan)
    route, query = _legacy_route(tmp_path)
    legacy = tmp_path / "legacy-alias"
    publish(root=legacy, queries=(query,), alias_routes=(route,))
    plan = json.loads((legacy / "plan.json").read_bytes())
    assert plan["alias_routes"] == [asdict(route)] and "schema" not in plan["alias_routes"][0]
    assert capture._alias_route_dict(route) == asdict(route)
    assert capture._alias_route_from_dict(asdict(route)) == route
    assert not _OPTIONAL & set(plan)


@pytest.mark.parametrize("mode", ["loose", "direct"])
@pytest.mark.parametrize("case", ["missing-tag", "wrong-tag", "missing-marker", "wrong-code-hash"])
def test_rehashed_bad_route_plan_rejects_before_started_or_http(originals, tmp_path, monkeypatch, mode, case):
    publish, acquire, _ = _api(mode)
    root = tmp_path / "capture"
    publish(root=root, queries=(_query(),), alias_routes=(_route(originals),))
    plan = json.loads((root / "plan.json").read_bytes())
    if case == "missing-tag":
        plan["alias_routes"][0].pop("schema")
    elif case == "wrong-tag":
        plan["alias_routes"][0]["schema"] = "unreviewed-route-v1"
    elif case == "missing-marker":
        plan.pop("documented_alias_schema")
    else:
        plan["documented_alias_implementation_sha256"] = "0" * 64
    changed = _replace_file(root / "plan.json", plan)
    before = _inventory(root)
    monkeypatch.setattr(transport, "_fetch", _denied)
    monkeypatch.setattr(transport.request, "build_opener", _denied)
    with pytest.raises(ValueError):
        acquire(root=root, plan_sha256=changed, api_key=KEY)
    assert _inventory(root) == before and not (root / "STARTED.json").exists()


@pytest.mark.parametrize("mode", ["loose", "direct"])
def test_changed_sec_original_blocks_planned_acquisition_without_request(originals, tmp_path, monkeypatch, mode):
    route = _copied_route(tmp_path / "evidence", originals)
    publish, acquire, _ = _api(mode)
    root = tmp_path / "capture"
    plan = publish(root=root, queries=(_query(),), alias_routes=(route,))
    Path(route.sec_source.path).write_bytes(b"different retained primary bytes")
    before = _inventory(root)
    monkeypatch.setattr(transport, "_fetch", _denied)
    monkeypatch.setattr(transport.request, "build_opener", _denied)
    with pytest.raises(ValueError):
        acquire(root=root, plan_sha256=plan, api_key=KEY)
    assert _inventory(root) == before and not (root / "STARTED.json").exists()


def test_corpus_identity_serialization_tags_only_documented_variant(originals, tmp_path):
    legacy, _ = _legacy_route(tmp_path)
    for identity in (corpus.CorpusIdentity("META", "FB", legacy.identity, legacy),
                     corpus.CorpusIdentity("AAPL", "AAPL", legacy.identity)):
        assert identity.to_dict() == {k: v for k, v in asdict(identity).items() if k != "reviewed_class"}
        assert set(identity.to_dict()) == {"fixed_slot_ticker", "provider_ticker", "identity", "alias", "successor"}
    route = _route(originals)
    identity = corpus.CorpusIdentity("BKNG", "PCLN", route.identity, route)
    assert identity.to_dict() == {**{k: v for k, v in asdict(identity).items() if k != "reviewed_class"},
                                  "alias": route.to_dict()}
    payload = json.loads(json.dumps(identity.to_dict()))
    restored = corpus.CorpusIdentity(payload["fixed_slot_ticker"], payload["provider_ticker"],
        corpus.AliasIdentityRef(**payload["identity"]), capture._alias_route_from_dict(payload["alias"]),
        payload["successor"])
    assert restored == identity
    module = "raw_second_documented_alias_v1.py"
    assert module not in corpus._implementation(corpus.DIRECT_STORAGE)
    assert corpus._implementation(corpus.DIRECT_STORAGE, alias_routing=corpus.DOCUMENTED_ALIAS_ROUTING)[module] == (
        sha256(Path(documented.__file__).read_bytes()).hexdigest())


@pytest.mark.parametrize("case", ["no-opt-in", "wrong-opt-in", "intended-path", "intended-hash",
                                   "slot", "outer-identity", "date", "mixed-successor"])
def test_corpus_exact_route_binding_rejects_before_any_write(originals, tmp_path, monkeypatch, case):
    route = _route(originals)
    plan = {"alias_routing": corpus.DOCUMENTED_ALIAS_ROUTING,
        "intended_issue_source": route.intended_issue_source,
        "intended_issue_sha256": route.intended_issue_sha256}
    batch = {"session_date": "2017-01-03", "start_ms": _query().start_ms, "end_ms": _query().end_ms,
        "fixed_slot_tickers": ["BKNG"], "diagnostic_only": True, "registry_batch_claimed": False}
    identity = corpus.CorpusIdentity("BKNG", "PCLN", route.identity, route)
    if case == "no-opt-in":
        plan.pop("alias_routing")
    elif case == "wrong-opt-in":
        plan["alias_routing"] = "unreviewed"
    elif case == "intended-path":
        plan["intended_issue_source"] += ".different"
    elif case == "intended-hash":
        plan["intended_issue_sha256"] = "0" * 64
    elif case == "slot":
        identity = replace(identity, fixed_slot_ticker="META")
    elif case == "outer-identity":
        identity = replace(identity, identity=route.modern_identity)
    elif case == "date":
        batch["session_date"] = "2017-01-04"
    else:
        identity = replace(identity, successor=object())
    monkeypatch.setattr(transport, "_fetch", _denied)
    monkeypatch.setattr(transport.request, "build_opener", _denied)
    with pytest.raises(ValueError):
        corpus._identity_queries(plan, batch, (identity,))
    assert not list(tmp_path.iterdir())


def _corpus_fixture(tmp_path, route, monkeypatch, enabled):
    # Retain the existing full-200/session registry fixture; change only its
    # intended-source binding before publishing any new corpus plan.
    registry, _ = _registry(tmp_path)
    summary = corpus._json(registry.root / "summary.json")
    summary["identity_evidence"]["sha256"] = route.intended_issue_sha256
    registry = _reseal_registry(registry, "summary.json", summary)
    monkeypatch.setattr(corpus, "validate_fixed_slot_identity", _identity)
    limits = corpus.CorpusLimits(2000, 768_000_000, 800_000_000, 700_000_000_000, tranche_batches=3)
    root, owner = tmp_path / "corpus", tmp_path / "owner"
    plan = corpus.publish_corpus_plan(root=root, registry=registry, limits=limits,
        intended_issue_source=Path(route.intended_issue_source), owner=owner,
        storage_mode=corpus.DIRECT_STORAGE, slot_accounting=corpus.SLOT_ACCOUNTING,
        alias_routing=corpus.DOCUMENTED_ALIAS_ROUTING if enabled else None)

    def identities(ordinal):
        batch = next(registry.batches(ordinal))
        return tuple(corpus.CorpusIdentity(name, "PCLN", route.identity, route) if name == "BKNG" else
            corpus.CorpusIdentity(name, name, route.modern_identity) for name in batch["fixed_slot_tickers"])

    def run(ordinal):
        return corpus.run_corpus_batch(root=root, plan_sha256=plan, identities=identities(ordinal),
            api_key="synthetic-test-only", capacity_snapshot=_capacity(project_ceiling_bytes=700_000_000_000))

    return root, plan, registry, limits, owner, identities, run


@pytest.mark.parametrize("enabled", [False, True])
def test_real_slot56_route_requires_opt_in_and_survives_persisted_prefix_replay(originals, tmp_path, monkeypatch, enabled):
    route = _copied_route(tmp_path / "evidence", originals)
    root, plan_sha, registry, limits, owner, identities, run = _corpus_fixture(tmp_path, route, monkeypatch, enabled)
    calls = _corpus_network(monkeypatch)
    source_before = _inventory(tmp_path / "evidence")
    plan = corpus._json(root / "plan.json")
    assert (plan.get("alias_routing") == corpus.DOCUMENTED_ALIAS_ROUTING) is enabled
    assert ("raw_second_documented_alias_v1.py" in plan["implementation_sha256"]) is enabled
    run(0)
    run(1)
    assert len(calls) == 48
    target = root / "batches/batch-000002"
    if not enabled:
        before = _inventory(root)
        with pytest.raises(ValueError, match="opt-in"):
            run(2)
        assert len(calls) == 48 and not target.exists() and _inventory(root) == before
        assert not (root / "COMPLETE.json").exists()
        return
    result = run(2)
    assert result["actual"]["http_requests"] == 24 and len(calls) == 72
    batch = next(registry.batches(2))
    assert batch["fixed_slot_tickers"] == list(corpus.SYMBOLS[48:72])
    started = corpus._json(target / "STARTED.json")
    bkng = started["identity_routes"][8]
    assert bkng == identities(2)[8].to_dict() and bkng["alias"]["schema"] == documented.SCHEMA
    assert started["identity_proofs"][8]["fixed_slot_index"] == 56
    assert started["identity_proofs"][8]["historical_figi_fields_absent"] is True
    capture_plan = corpus._json(target / "capture/plan.json")
    assert capture_plan["queries"][8]["ticker"] == "PCLN"
    assert capture_plan["alias_routes"] == [route.to_dict()]
    assert result["slot_accounting"]["slots"][8] == dict(fixed_slot_ticker="BKNG", fixed_slot_index=56,
        disposition="acquired", query=asdict(_query()))
    before = _inventory(root)
    monkeypatch.setattr(transport, "_fetch", _denied)
    monkeypatch.setattr(transport.request, "build_opener", _denied)
    complete = run(2)
    assert complete["range_complete"] and complete["next_batch"] == 3
    assert complete["slot_totals"]["slot_count"] == 72
    assert _inventory(root) == before and _inventory(tmp_path / "evidence") == source_before
    # Removing the explicit marker on a new successor tranche cannot promote
    # this completed opt-in prefix into a default contract or reserve an owner.
    previous = corpus.CaptureReuse(root, plan_sha, sha256((root / "COMPLETE.json").read_bytes()).hexdigest(),
                                   corpus.DIRECT_STORAGE)
    owner_before = _inventory(owner)
    with pytest.raises(ValueError, match="Predecessor"):
        corpus.publish_corpus_plan(root=tmp_path / "wrong-successor", registry=registry, limits=limits,
            intended_issue_source=Path(route.intended_issue_source), owner=owner, previous=previous,
            storage_mode=corpus.DIRECT_STORAGE, slot_accounting=corpus.SLOT_ACCOUNTING)
    assert not (tmp_path / "wrong-successor").exists() and _inventory(owner) == owner_before
    # Completed-prefix replay must reopen the actual SEC original, not trust
    # the previously persisted proof or merely the unchanged capture index.
    Path(route.sec_source.path).write_bytes(b"changed after completed capture")
    with pytest.raises(ValueError):
        run(2)
    assert _inventory(root) == before and len(calls) == 72


@pytest.mark.parametrize("case", ["complete-route", "route-only", "marker-only", "code-hash-only"])
def test_mixed_reuse_requires_opt_in_before_verifier_or_batch_started(originals, tmp_path, monkeypatch, case):
    route, query = _route(originals), _query()
    reusable = tmp_path / "mixed-capture"
    plan_sha = direct.publish_direct_second_capture_plan(root=reusable,
        queries=(replace(query, ticker="AAPL"), query), alias_routes=(route,))
    calls = _corpus_network(monkeypatch)
    direct.capture_direct_seconds(root=reusable, plan_sha256=plan_sha, api_key="synthetic-test-only")
    complete_sha = sha256((reusable / "COMPLETE.json").read_bytes()).hexdigest()
    ref = corpus.CaptureReuse(reusable, plan_sha, complete_sha, corpus.DIRECT_STORAGE)
    keys, proof = corpus._corpus_reuse(ref, {"alias_routing": corpus.DOCUMENTED_ALIAS_ROUTING})
    assert keys == {(ticker, query.start_ms, query.end_ms) for ticker in ("AAPL", "PCLN")}
    assert proof["query_count"] == len(calls) == 2
    # Corrupted marker-only copies are intentionally not native-complete;
    # even their attempted replay must require semantic admission beforehand.
    if case != "complete-route":
        changed = corpus._json(reusable / "plan.json")
        if case == "route-only":
            changed.pop("documented_alias_schema")
            changed.pop("documented_alias_implementation_sha256")
        else:
            changed["alias_routes"] = []
            changed.pop("documented_alias_implementation_sha256" if case == "marker-only"
                        else "documented_alias_schema")
        ref = replace(ref, plan_sha256=_replace_file(reusable / "plan.json", changed))
    root, plan, _, _, _, identities, _ = _corpus_fixture(tmp_path, route, monkeypatch, enabled=False)
    before = _inventory(tmp_path)

    def denied_verifier(self):
        pytest.fail("Off-batch documented PCLN evidence was replayed before corpus opt-in")

    monkeypatch.setattr(corpus.CaptureReuse, "verify", denied_verifier)
    monkeypatch.setattr(transport, "_fetch", _denied)
    with pytest.raises(ValueError, match="routing opt-in"):
        corpus.run_corpus_batch(root=root, plan_sha256=plan, identities=identities(0), reuse=(ref,),
            api_key="synthetic-test-only", capacity_snapshot=_capacity(project_ceiling_bytes=700_000_000_000))
    assert not list((root / "batches").iterdir())
    assert _inventory(tmp_path) == before and len(calls) == 2


_FRESH_REPLAY = r"""
import json, sys
from pathlib import Path
from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive import raw_second_capture_v1 as loose
from rl_quant.data_sources.massive import raw_second_direct_capture_v1 as direct
from rl_quant.data_sources.massive.raw_second_evidence_v1 import EvidenceRelocation
def denied(*args, **kwargs):
    raise AssertionError('Fresh source replay attempted provider transport')
transport._fetch = denied
transport.request.build_opener = denied
resolver = EvidenceRelocation(**json.loads(sys.argv[1]))
replay = loose.replay_second_capture if sys.argv[2] == 'loose' else direct.replay_direct_second_capture
result = replay(root=Path(resolver.package_root) / 'capture', plan_sha256=resolver.plan_sha256,
                completion_sha256=resolver.completion_sha256, evidence_relocation=resolver)
assert result['page_count'] == 1 and result['training_ready'] is False
assert result['evidence_relocation']['exact_bytes_verified'] is True
assert not any(name.startswith(('rl_quant.models.', 'rl_quant.training.')) for name in sys.modules)
print(json.dumps(result, sort_keys=True))
"""


@pytest.mark.parametrize("mode", ["loose", "direct"])
def test_fresh_source_only_replay_uses_explicit_mapping_without_original_locations(originals, tmp_path, monkeypatch, mode):
    original = tmp_path / "original"
    route = _copied_route(original, originals)
    query, root = _query(), original / "capture"
    publish, acquire, replay = _api(mode)
    plan = publish(root=root, queries=(query,), alias_routes=(route,))
    calls = _network(monkeypatch, [(query.url, 200, _body(query))])
    acquire(root=root, plan_sha256=plan, api_key=KEY)
    complete = sha256((root / "COMPLETE.json").read_bytes()).hexdigest()
    package, mapping = tmp_path / "package", []
    for path in sorted(original.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(original).as_posix()
        name = relative if relative.startswith("capture/") else "evidence/" + relative
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = path.read_bytes()
        target.write_bytes(raw)
        mapping.append(dict(original_path=str(path), published_path=name, bytes=len(raw),
                            sha256=sha256(raw).hexdigest()))
    resolver = _seal(package, mapping, plan, complete)
    original.rename(tmp_path / "retained-original")  # Preserve bytes; remove only old test-owned locations.
    before = _inventory(tmp_path)
    monkeypatch.setattr(transport, "_fetch", _denied)
    monkeypatch.setattr(transport.request, "build_opener", _denied)
    with pytest.raises(OSError):
        replay(root=package / "capture", plan_sha256=plan, completion_sha256=complete)
    expected = replay(root=package / "capture", plan_sha256=plan, completion_sha256=complete,
                      evidence_relocation=resolver)
    # Child is source-I/O only: it does not create a CUDA context, model,
    # optimizer, or tensor. Numerical fresh-process acceptance remains separate.
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    child = subprocess.run([sys.executable, "-B", "-c", _FRESH_REPLAY, json.dumps(asdict(resolver)), mode],
        env=environment, capture_output=True, text=True, timeout=180)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == expected
    assert _inventory(tmp_path) == before and len(calls) == 1
