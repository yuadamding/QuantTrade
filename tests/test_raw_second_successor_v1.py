"""LSF-only AVGO route tests against staged original evidence.

QT200_SUCCESSOR_EVIDENCE_ROOT is mandatory. The acceptance package must stage
the five exact files below; absence is an error, never a skip or synthetic
replacement. Negative cases mutate copies, not source originals or trusted
source pins. No market-data request, tensor/model update, or ledger conversion
is needed to validate this acquisition-only relationship.
"""

import base64
from dataclasses import asdict, replace
from datetime import datetime
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import torch

from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive import raw_second_corpus_v1 as corpus
from rl_quant.data_sources.massive import raw_second_direct_capture_v1 as direct
from rl_quant.data_sources.massive.raw_second_alias_v1 import AliasIdentityRef
from rl_quant.data_sources.massive.raw_second_successor_v1 import (
    ReviewedSuccessorSourceRef, SecondSuccessorRoute, _query_clock,
)
from rl_quant.datasets.massive_raw_seconds_v1 import SecondQuery
from rl_quant.datasets.massive_raw_second_packed_v1 import PackedSecondCaptureRef
from test_raw_second_corpus_v1 import _capacity, _network, _registry

pytestmark = pytest.mark.lsf_gpu
_ORIGINALS = {
    "avgo-sec-20180404.html":
        "dd5a6bf29b945d982449a0b471ad685257f80c71fdb2b220919f721b0053339e",
    "avgo-nasdaq-20180405.html":
        "fa36bd667d907a9ce09f0c33499117b7fd7ef3608c68eb77f2d087ee320d8d49",
    "avgo-historical-20170103.json":
        "c0cbe6af3eb7407ee05483329ce88720a30d2c78548fc82379aa6d231d57e30f",
    "avgo-successor-20220103.json":
        "9a448517abea29578aa3264e30edd9d85b1e3b2d935c3cb347ed811987b8ef10",
    "intended-issues.json.gz":
        "9f0ef1e22cc939a09302a6bb8970687a7e4e8fee697db9d590c9e710fe0194c0",
}
_EASTERN = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True, scope="module")
def assigned_gpu():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1


@pytest.fixture(scope="module")
def originals():
    root = Path(os.environ["QT200_SUCCESSOR_EVIDENCE_ROOT"])
    result = {}
    for name, expected in _ORIGINALS.items():
        raw = transport.read_regular(root / name, 8 * 1024 * 1024)
        assert sha256(raw).hexdigest() == expected, name
        result[name] = raw
    return result


def _query(day="2017-01-03", clock="09:30:00", seconds=3600):
    start = int(datetime.fromisoformat(day + "T" + clock).replace(tzinfo=_EASTERN).timestamp()) * 1000
    return SecondQuery("AVGO", start, start + (seconds - 1) * 1000)


def _route(tmp_path, originals):
    for name, raw in originals.items():
        (tmp_path / name).write_bytes(raw)
    route = SecondSuccessorRoute(
        fixed_slot_ticker="AVGO", provider_ticker="AVGO", session_date="2017-01-03",
        identity=AliasIdentityRef("dated-ticker-wrapper",
            str(tmp_path / "avgo-historical-20170103.json"), _ORIGINALS["avgo-historical-20170103.json"]),
        successor_identity=AliasIdentityRef("dated-ticker-wrapper",
            str(tmp_path / "avgo-successor-20220103.json"), _ORIGINALS["avgo-successor-20220103.json"]),
        successor_observation_date="2022-01-03",
        intended_issue_source=str(tmp_path / "intended-issues.json.gz"),
        intended_issue_sha256=_ORIGINALS["intended-issues.json.gz"],
        sec_source=ReviewedSuccessorSourceRef("sec-20180404",
            str(tmp_path / "avgo-sec-20180404.html"), _ORIGINALS["avgo-sec-20180404.html"]),
        nasdaq_source=ReviewedSuccessorSourceRef("nasdaq-20180405",
            str(tmp_path / "avgo-nasdaq-20180405.html"), _ORIGINALS["avgo-nasdaq-20180405.html"]),
    )
    return route, _query()


def _inventory(root):
    return {name: sha256((root / name).read_bytes()).hexdigest() for name in _ORIGINALS}


def _rewrite_identity(reference, updates):
    """Rehash a deliberately false provider-shaped copy; never trusted HTML."""
    path = Path(reference.path)
    wrapper = json.loads(path.read_bytes())
    body = json.loads(base64.b64decode(wrapper["raw_response_body_base64"]))
    body["results"].update(updates)
    raw = json.dumps(body, sort_keys=True).encode()
    wrapper.update(results=body["results"], raw_response_body_base64=base64.b64encode(raw).decode(),
                   raw_response_body_sha256=sha256(raw).hexdigest(), raw_response_content_length=len(raw))
    raw = json.dumps(wrapper, sort_keys=True).encode()
    path.write_bytes(raw)
    return replace(reference, sha256=sha256(raw).hexdigest())


def test_actual_originals_resolve_only_acquisition_and_preserve_old_identifiers(tmp_path, originals, monkeypatch):
    route, query = _route(tmp_path, originals)
    before = _inventory(tmp_path)

    def no_network(*args, **kwargs):
        raise AssertionError("Identity validation must not acquire provider data")

    monkeypatch.setattr(transport, "_fetch", no_network)
    result = route.validate(query)
    assert result["historical_provider_cik"] == "0001649338"
    assert result["historical_provider_composite_figi"] is None
    assert result["historical_provider_share_class_figi"] is None
    assert result["intended_cik"] == "0001730168"
    assert result["intended_composite_figi"] == "BBG00KHY5S69"
    assert result["intended_share_class_figi"] == "BBG00KHY5SY8"
    assert result["provider_ticker"] == result["fixed_slot_ticker"] == "AVGO"
    assert result["original_public_cusip"] == "Y09827109"
    assert result["successor_public_cusip"] == "11135F101"
    assert result["exchange_ratio_numerator"] == result["exchange_ratio_denominator"] == 1
    assert result["legal_effective_boundary"] == "after-market-close-2018-04-04"
    assert result["legal_effective_timestamp_ms"] is None
    assert result["marketplace_effective_date"] == "2018-04-05"
    assert result["acquisition_route_only"] is True
    for field in (
        "provider_identifiers_rewritten", "provider_tickers_rewritten",
        "query_clock_is_exchange_calendar_qualification", "continuous_identity_qualified",
        "historical_issue_identity_qualified", "economic_conversion_qualified",
        "economic_accounting_qualified", "tradability_qualified", "point_in_time_qualified",
        "training_ready",
    ):
        assert result[field] is False
    restored = SecondSuccessorRoute.from_dict(json.loads(json.dumps(route.to_dict())))
    assert restored == route and restored.validate(query) == result
    # The same exact dated observation supports another hour, not another day.
    assert restored.validate(_query(clock="14:30:00")) == result
    assert _inventory(tmp_path) == before == _ORIGINALS


@pytest.mark.parametrize("case", ["changed-html", "matching-keywords", "report-copy", "wrong-hash",
                                   "swapped-kind", "symlink", "hardlink"])
def test_public_evidence_requires_exact_reviewed_original_not_a_new_hash(tmp_path, originals, case):
    route, query = _route(tmp_path, originals)
    source = route.sec_source
    path = Path(source.path)
    if case in ("changed-html", "matching-keywords", "report-copy"):
        before = path.read_bytes()
        raw = {
            "changed-html": before.replace(b"one-for-one", b"two-for-one", 1),
            "matching-keywords": b"Broadcom Limited AVGO one-for-one April 4 2018 April 5 2018",
            "report-copy": json.dumps({"approved": True, "ratio": 1, "old_cik": "0001649338"}).encode(),
        }[case]
        assert raw != before
        path.write_bytes(raw)
        source = replace(source, sha256=sha256(raw).hexdigest())
    elif case == "wrong-hash":
        source = replace(source, sha256="0" * 64)
    elif case == "swapped-kind":
        source = route.nasdaq_source
    else:
        linked = tmp_path / "linked-source.html"
        if case == "symlink":
            linked.symlink_to(path)
        else:
            linked.hardlink_to(path)
        source = replace(source, path=str(linked))
    with pytest.raises(ValueError):
        replace(route, sec_source=source).validate(query)


@pytest.mark.parametrize("field,value", [
    ("cik", "0001730168"), ("name", "Broadcom Cayman L.P."),
    ("composite_figi", "BBG00KHY5S69"), ("share_class_figi", "BBG00KHY5SY8"),
    ("primary_exchange", "XNYS"), ("type", "PFD"), ("currency_name", "sgd"),
    ("active", False), ("list_date", "2017-01-04"), ("ticker", "BRCM"),
    ("market", "crypto"), ("locale", "sg"),
])
def test_missing_figi_exception_cannot_admit_another_historical_issue(tmp_path, originals, field, value):
    route, query = _route(tmp_path, originals)
    identity = _rewrite_identity(route.identity, {field: value})
    with pytest.raises(ValueError):
        replace(route, identity=identity).validate(query)


@pytest.mark.parametrize("field,value", [
    ("cik", "0001649338"), ("name", "Broadcom Limited Ordinary Shares"),
    ("composite_figi", None), ("share_class_figi", "DIFFERENT"),
    ("ticker", "XOM"), ("type", "ETF"),
])
def test_successor_must_be_reauthenticated_from_its_original_provider_body(tmp_path, originals, field, value):
    route, query = _route(tmp_path, originals)
    identity = _rewrite_identity(route.successor_identity, {field: value})
    with pytest.raises(ValueError):
        replace(route, successor_identity=identity).validate(query)


@pytest.mark.parametrize("case", ["wrong-slot", "wrong-ticker", "other-day", "pre-scope",
                                   "market-effective-date", "pre-open", "post-close", "over-hour",
                                   "midnight", "pre-exchange-successor"])
def test_successor_is_exact_day_bounded_clock_not_continuous_identity(tmp_path, originals, case):
    route, query = _route(tmp_path, originals)
    if case == "wrong-slot":
        route = replace(route, fixed_slot_ticker="XOM")
    elif case == "wrong-ticker":
        query = replace(query, ticker="BRCM")
    elif case == "other-day":
        route, query = replace(route, session_date="2017-01-04"), _query("2017-01-04")
    elif case == "pre-scope":
        route, query = replace(route, session_date="2017-01-02"), _query("2017-01-02")
    elif case == "market-effective-date":
        route, query = replace(route, session_date="2018-04-05"), _query("2018-04-05")
    elif case == "pre-open":
        query = _query(clock="09:29:59", seconds=2)
    elif case == "post-close":
        route = replace(route, session_date="2018-04-04")
        query = _query("2018-04-04", "15:59:59", seconds=2)
    elif case == "over-hour":
        query = _query(seconds=3601)
    elif case == "midnight":
        query = _query(clock="23:59:59", seconds=2)
    else:
        route = replace(route, successor_observation_date="2018-04-04")
    with pytest.raises(ValueError):
        route.validate(query)


@pytest.mark.parametrize("day,clock", [("2017-01-03", "09:30:00"), ("2018-04-04", "15:59:59")])
def test_clock_boundary_alone_does_not_claim_a_dated_provider_observation(day, clock):
    _query_clock(_query(day, clock, seconds=1), day)


@pytest.mark.parametrize("stamp", [True, 1000.0, 1001])
def test_unsafe_query_deserialization_cannot_bypass_integral_second_contract(tmp_path, originals, stamp):
    route, query = _route(tmp_path, originals)
    object.__setattr__(query, "start_ms", stamp)
    with pytest.raises(ValueError):
        route.validate(query)


@pytest.mark.parametrize("case", ["outer-results", "raw-sha", "request-url", "copied-report"])
def test_original_wrapper_and_raw_body_chain_remain_required(tmp_path, originals, case):
    route, query = _route(tmp_path, originals)
    path = Path(route.identity.path)
    body = json.loads(path.read_bytes())
    if case == "outer-results":
        body["results"]["cik"] = "0001730168"
    elif case == "raw-sha":
        body["raw_response_body_sha256"] = "0" * 64
    elif case == "request-url":
        body["request_url"] = body["request_url"].replace("2017-01-03", "2017-01-04")
    else:
        body.pop("raw_response_body_base64")
    raw = json.dumps(body).encode()
    path.write_bytes(raw)
    identity = replace(route.identity, sha256=sha256(raw).hexdigest())
    with pytest.raises((ValueError, KeyError)):
        replace(route, identity=identity).validate(query)


@pytest.mark.parametrize("case", ["other-cik", "other-figi", "missing-slot", "wrong-order", "boolean-count"])
def test_rehashed_intended_report_cannot_change_reviewed_issue_or_fixed_population(tmp_path, originals, case):
    route, query = _route(tmp_path, originals)
    path = Path(route.intended_issue_source)
    body = json.loads(gzip.decompress(path.read_bytes()))
    index = next(i for i, row in enumerate(body["security_resolutions"]) if row["qt200_ticker"] == "AVGO")
    intended = body["security_resolutions"][index]["current_reference_evidence"]
    if case == "other-cik":
        intended["current_issuer_cik"] = "0001649338"
    elif case == "other-figi":
        intended["current_composite_figi"] = "OTHER"
    elif case == "missing-slot":
        body["security_resolutions"].pop(index)
    elif case == "wrong-order":
        body["ordered_tickers"].reverse()
    else:
        intended["active_exact_reference_count"] = True
    raw = gzip.compress(json.dumps(body).encode(), mtime=0)
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        replace(route, intended_issue_sha256=sha256(raw).hexdigest()).validate(query)


@pytest.mark.parametrize("case", ["extra-mapping", "missing-source", "extra-source-field", "duck-source"])
def test_serialization_does_not_accept_free_successor_mappings(tmp_path, originals, case):
    route, query = _route(tmp_path, originals)
    body = route.to_dict()
    if case == "extra-mapping":
        body["exchange_ratio"] = 2
    elif case == "missing-source":
        body.pop("nasdaq_source")
    elif case == "extra-source-field":
        body["sec_source"]["approved"] = True
    else:
        with pytest.raises(ValueError):
            replace(route, sec_source=body["sec_source"]).validate(query)
        return
    with pytest.raises(ValueError):
        SecondSuccessorRoute.from_dict(body)


def _corpus_setup(tmp_path, originals, monkeypatch):
    route, query = _route(tmp_path, originals)
    calendar_root = tmp_path / "synthetic-calendar"
    calendar_root.mkdir()
    original_registry, _ = _registry(calendar_root)
    # Reuse only the existing synthetic calendar geometry. The successor
    # identity table and AVGO qualifier remain actual, original, and unpatched.
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    files = []
    for name in ("summary.json", "universe.json", "sessions.jsonl", "reuse-queries.jsonl"):
        raw = (original_registry.root / name).read_bytes()
        if name == "summary.json":
            body = json.loads(raw)
            body["identity_evidence"] = {"sha256": route.intended_issue_sha256}
            raw = transport.canonical(body)
        files.append(transport.write_once(registry_root / name, raw))
    complete = transport.write_once(registry_root / "COMPLETE.json", transport.canonical(
        {"metadata_complete": True, "files": files}))["sha256"]
    registry = corpus.CorpusRegistry(registry_root, complete)

    def other_fixture_slots_only(**kwargs):
        assert kwargs["fixed_slot_ticker"] != "AVGO", "AVGO cannot use the ordinary-identity substitute"
        return dict(fixed_slot_ticker=kwargs["fixed_slot_ticker"],
                    session_date=kwargs["session_date"], training_ready=False)

    monkeypatch.setattr(corpus, "validate_fixed_slot_identity", other_fixture_slots_only)
    limits = corpus.CorpusLimits(maximum_http_requests=200, maximum_raw_response_bytes=128_000_000,
        maximum_retained_bytes=200_000_000, project_ceiling_bytes=700_000_000_000, tranche_batches=1)
    root = tmp_path / "corpus"
    plan_sha = corpus.publish_corpus_plan(root=root, registry=registry, limits=limits,
        intended_issue_source=Path(route.intended_issue_source), owner=tmp_path / "owner",
        storage_mode=corpus.DIRECT_STORAGE)
    batch = next(registry.batches())
    assert len(batch["fixed_slot_tickers"]) == 24 and "AVGO" in batch["fixed_slot_tickers"]
    assert (batch["start_ms"], batch["end_ms"]) == (query.start_ms, query.end_ms)
    identities = tuple(corpus.CorpusIdentity(ticker, ticker, route.identity,
        successor=route if ticker == "AVGO" else None) for ticker in batch["fixed_slot_tickers"])
    return root, plan_sha, identities, route


def _run_corpus(setup):
    root, plan_sha, identities, _ = setup
    return corpus.run_corpus_batch(root=root, plan_sha256=plan_sha, identities=identities,
        api_key="synthetic-transport-only", capacity_snapshot=_capacity(project_ceiling_bytes=700_000_000_000))


def test_explicit_successor_query_never_calls_ordinary_identity_validator(tmp_path, originals, monkeypatch):
    route, query = _route(tmp_path, originals)

    def forbidden_ordinary(**kwargs):
        raise AssertionError("Successor evidence must not be replaced by ordinary FIGI equality")

    monkeypatch.setattr(corpus, "validate_fixed_slot_identity", forbidden_ordinary)
    identity = corpus.CorpusIdentity("AVGO", "AVGO", route.identity, successor=route)
    serialized = json.loads(json.dumps(asdict(identity)))
    restored = corpus.CorpusIdentity(serialized["fixed_slot_ticker"], serialized["provider_ticker"],
        AliasIdentityRef(**serialized["identity"]), alias=None,
        successor=SecondSuccessorRoute.from_dict(serialized["successor"]))
    assert restored == identity
    plan = dict(intended_issue_source=route.intended_issue_source,
                intended_issue_sha256=route.intended_issue_sha256)
    batch = dict(fixed_slot_tickers=["AVGO"], session_date=route.session_date,
                 start_ms=query.start_ms, end_ms=query.end_ms)
    queries, aliases, proofs = corpus._identity_queries(plan, batch, (restored,))
    assert queries == [query] and aliases == [] and proofs == [route.validate(query)]


@pytest.mark.parametrize("case", ["alias", "date", "identity", "intended-path", "intended-sha",
                                   "slot", "provider"])
def test_corpus_successor_mismatch_stops_before_http_or_batch_directory(tmp_path, originals, monkeypatch, case):
    root, plan_sha, identities, route = _corpus_setup(tmp_path, originals, monkeypatch)
    calls = _network(monkeypatch)
    index = next(i for i, identity in enumerate(identities) if identity.fixed_slot_ticker == "AVGO")
    item = identities[index]
    if case == "alias":
        item = replace(item, alias="unsupported-simultaneous-alias")
    elif case == "date":
        item = replace(item, successor=replace(route, session_date="2017-01-04"))
    elif case == "identity":
        item = replace(item, identity=replace(route.identity, sha256="f" * 64))
    elif case == "intended-path":
        item = replace(item, successor=replace(route, intended_issue_source=route.intended_issue_source + ".other"))
    elif case == "intended-sha":
        item = replace(item, successor=replace(route, intended_issue_sha256="0" * 64))
    elif case == "slot":
        item = replace(item, successor=replace(route, fixed_slot_ticker="AAPL"))
    else:
        item = replace(item, successor=replace(route, provider_ticker="BRCM"))
    identities = identities[:index] + (item,) + identities[index + 1:]
    with pytest.raises(ValueError, match="bound to this exact corpus"):
        _run_corpus((root, plan_sha, identities, route))
    assert calls == [] and not list((root / "batches").iterdir())


def test_persisted_direct_corpus_replays_original_successor_proof_and_blocks_corruption(tmp_path, originals, monkeypatch):
    setup = _corpus_setup(tmp_path, originals, monkeypatch)
    calls = _network(monkeypatch)
    result = _run_corpus(setup)
    assert result["batch_complete"] is True and result["actual"]["http_requests"] == len(calls) == 24
    root, _, identities, route = setup
    target = root / "batches" / "batch-000000"
    started = json.loads((target / "STARTED.json").read_bytes())
    index = next(i for i, identity in enumerate(identities) if identity.fixed_slot_ticker == "AVGO")
    assert started["identity_routes"][index] == {
        k: v for k, v in asdict(identities[index]).items() if k != "reviewed_class"}
    assert "reviewed_class" not in started["identity_routes"][index]
    assert started["identity_proofs"][index] == route.validate(_query())
    terminal = _run_corpus(setup)
    assert terminal["range_complete"] is True and terminal["training_ready"] is False
    assert len(calls) == 24
    native = direct.replay_direct_second_capture(root=target / "capture",
        plan_sha256=result["capture"]["plan_sha256"],
        completion_sha256=result["capture"]["completion_sha256"])
    references = [PackedSecondCaptureRef.from_dict(value) for value in native["captures"]]
    avgo = next(reference for reference in references if reference.query().ticker == "AVGO")
    query, pages = avgo.load()
    assert query == _query() and len(pages) == 1
    assert json.loads(pages[0].body)["results"] == [
        {"t": query.start_ms, "o": 100, "h": 101, "l": 99, "c": 100, "v": 10}]
    assert native["training_ready"] is False and len(calls) == 24
    source = Path(route.sec_source.path)
    before = source.read_bytes()
    changed = before.replace(b"one-for-one", b"two-for-one", 1)
    assert changed != before
    source.write_bytes(changed)
    with pytest.raises(ValueError, match="primary bytes changed"):
        _run_corpus(setup)
    assert len(calls) == 24, "Corrupt completed-prefix identity evidence must not trigger reacquisition"
