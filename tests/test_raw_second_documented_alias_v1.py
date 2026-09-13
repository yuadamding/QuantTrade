"""LSF-only documented BKNG/PCLN route tests against retained originals.

QT200_DOCUMENTED_ALIAS_EVIDENCE_ROOT must contain all nine files in _ORIGINALS.
The historical subdirectory preserves the original supplemental receipt chain;
the SEC basename matches its original receipt. Missing fixtures fail, never
skip. Negative tests mutate only test-owned copies and never replace trusted
source pins. No provider request, market feature, model, or ledger is needed.
"""

import base64
from dataclasses import replace
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
from rl_quant.data_sources.massive.raw_second_alias_v1 import (
    AliasEventRef, AliasIdentityRef, SecondAliasRoute,
)
from rl_quant.data_sources.massive.raw_second_documented_alias_v1 import (
    SCHEMA, ReviewedDocumentedAliasSourceRef, SecondDocumentedAliasRoute,
)
from rl_quant.data_sources.massive.raw_second_evidence_v1 import EvidenceRelocation, MAP_NAME
from rl_quant.datasets.massive_raw_seconds_v1 import SecondQuery

pytestmark = pytest.mark.lsf_gpu
_EASTERN = ZoneInfo("America/New_York")
_ORIGINALS = {
    "historical/COMPLETE.json":
        "50c403b3bce22e6a17cde2a3f7f95247472fc479cd348b993d3b74b807dda820",
    "historical/page-0000.receipt.json":
        "d1643c9849758d7577dd3d0e767ed5ca4f46ba5fa52b8160c8d325f5c9991ce2",
    "historical/page-0000.json.gz":
        "ac9c238bd7b6b6670a278b59fa46f0950e86c4aab7c658015200def738c743f2",
    "bkng-sec-20180227.html":
        "5ab36122e9379cac575e408b2ac6610ecac485c9be5f483eaacb40c798ea156b",
    "bkng-sec-20180227.receipt.json":
        "0b5ff9e7588cceebbb267142eaec360bb87d37df7823d1c9dc3e818e1f3abf48",
    "bkng-modern-20220103.json":
        "e8234d36868911569c8a14dd5872f33dd92b1d1b92092355e64af07b6c0d0b76",
    "intended-issues.json.gz":
        "9f0ef1e22cc939a09302a6bb8970687a7e4e8fee697db9d590c9e710fe0194c0",
    "bkng-404-20170103.json":
        "99d120a236accbb2e2b570062097d4cec235d1a52f5ee3ee3d9e585170e2094a",
    "ticker-events-v1.json":
        "19baad0908fcfd350e3221625a2f776c38e9b462baeb7ae4c5d049fc88df4f8e",
}
_ROUTE_FILES = tuple(name for name in _ORIGINALS
                     if name not in ("bkng-404-20170103.json", "ticker-events-v1.json"))
_FLAGS = dict(training_ready=False, training_authorized=False,
              remote_native_replay_qualified=False, native_catalog_activated=False,
              original_references_rewritten=False)


@pytest.fixture(autouse=True, scope="module")
def assigned_gpu():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1


@pytest.fixture(scope="module")
def originals():
    root = Path(os.environ["QT200_DOCUMENTED_ALIAS_EVIDENCE_ROOT"])
    for name, expected in _ORIGINALS.items():
        assert sha256(transport.read_regular(root / name, 8 * 1024**2)).hexdigest() == expected, name
    return root


def _query(day="2017-01-03", clock="09:30:00", seconds=3600):
    start = int(datetime.fromisoformat(day + "T" + clock).replace(tzinfo=_EASTERN).timestamp()) * 1000
    return SecondQuery("PCLN", start, start + (seconds - 1) * 1000)


def _route(root):
    return SecondDocumentedAliasRoute(
        fixed_slot_ticker="BKNG", provider_ticker="PCLN", session_date="2017-01-03",
        identity=AliasIdentityRef("reference-supplement", str(root / "historical"),
                                  _ORIGINALS["historical/COMPLETE.json"]),
        modern_identity=AliasIdentityRef("dated-ticker-wrapper", str(root / "bkng-modern-20220103.json"),
                                         _ORIGINALS["bkng-modern-20220103.json"]),
        modern_observation_date="2022-01-03", intended_issue_source=str(root / "intended-issues.json.gz"),
        intended_issue_sha256=_ORIGINALS["intended-issues.json.gz"],
        sec_source=ReviewedDocumentedAliasSourceRef("bkng-sec-20180227",
            str(root / "bkng-sec-20180227.html"), _ORIGINALS["bkng-sec-20180227.html"],
            str(root / "bkng-sec-20180227.receipt.json"), _ORIGINALS["bkng-sec-20180227.receipt.json"]),
    )


def _copied_route(root, originals):
    root.mkdir(parents=True, exist_ok=True)
    for name in _ROUTE_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((originals / name).read_bytes())
    return _route(root)


def _inventory(root):
    return {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _denied(*args, **kwargs):
    raise AssertionError("Documented route validation must not request provider data")


def _rehash_historical(route, *, updates=None, duplicate=False, empty=False):
    """Make an internally consistent false copy, not a new trusted original."""
    root = Path(route.identity.path)
    payload = json.loads(gzip.decompress((root / "page-0000.json.gz").read_bytes()))
    if updates:
        payload["results"][0].update(updates)
    if duplicate:
        payload["results"].append(dict(payload["results"][0]))
    if empty:
        payload["results"] = []
    payload["count"] = len(payload["results"])
    raw = transport.canonical(payload)
    packed = gzip.compress(raw, mtime=0)
    (root / "page-0000.json.gz").write_bytes(packed)
    receipt = json.loads((root / "page-0000.receipt.json").read_bytes())
    receipt.update(raw_body_bytes=len(raw), raw_body_sha256=sha256(raw).hexdigest())
    receipt["body"].update(bytes=len(packed), sha256=sha256(packed).hexdigest())
    body = transport.canonical(receipt)
    (root / "page-0000.receipt.json").write_bytes(body)
    complete = json.loads((root / "COMPLETE.json").read_bytes())
    complete["pages"][0].update(bytes=len(body), sha256=sha256(body).hexdigest())
    complete["result_count"] = len(payload["results"])
    body = transport.canonical(complete)
    (root / "COMPLETE.json").write_bytes(body)
    return replace(route, identity=replace(route.identity, sha256=sha256(body).hexdigest()))


def _rehash_modern(route, updates):
    path = Path(route.modern_identity.path)
    wrapper = json.loads(path.read_bytes())
    payload = json.loads(base64.b64decode(wrapper["raw_response_body_base64"], validate=True))
    payload["results"].update(updates)
    raw = transport.canonical(payload)
    wrapper.update(results=payload["results"], raw_response_body_base64=base64.b64encode(raw).decode(),
                   raw_response_body_sha256=sha256(raw).hexdigest(), raw_response_content_length=len(raw))
    body = transport.canonical(wrapper)
    path.write_bytes(body)
    return replace(route, modern_identity=replace(route.modern_identity, sha256=sha256(body).hexdigest()))


def test_retained_originals_preserve_literal_ticker_missing_ids_and_fixed_axis(originals, monkeypatch):
    route, query = _route(originals), _query()
    before = _inventory(originals)
    monkeypatch.setattr(transport, "_fetch", _denied)
    monkeypatch.setattr(transport.request, "build_opener", _denied)
    rows = route.identity.rows(fixed_slot="BKNG", provider_ticker="PCLN", session_date="2017-01-03")
    assert len(rows) == 1 and rows[0]["ticker"] == "PCLN"
    assert rows[0]["name"] == "The Priceline Group Inc." and rows[0]["cik"] == "0001075531"
    assert "composite_figi" not in rows[0] and "share_class_figi" not in rows[0]
    proof = route.validate(query)
    assert proof["fixed_slot_ticker"] == "BKNG" and proof["provider_ticker"] == "PCLN"
    intended = json.loads(gzip.decompress((originals / "intended-issues.json.gz").read_bytes()))
    assert proof["fixed_slot_index"] == 56 and len(intended["ordered_tickers"]) == 200
    assert intended["ordered_tickers"][56] == "BKNG"
    assert proof["ordered_tickers_sha256"] == "92002983b9b6686bf8b6f5d6f88d212cf97f61f1f98bb55d49f262fa33b491fe"
    assert proof["historical_provider_name"] == "The Priceline Group Inc."
    assert proof["historical_provider_cik"] == proof["intended_cik"] == "0001075531"
    assert proof["historical_provider_composite_figi"] is None
    assert proof["historical_provider_share_class_figi"] is None
    assert proof["historical_figi_fields_absent"] is True
    assert proof["intended_composite_figi"] == "BBG000BLBVN4"
    assert proof["intended_share_class_figi"] == "BBG001S89N72"
    assert proof["security_class"] == "common stock"
    assert proof["relationship"] == "documented-continuing-common-stock-name-ticker-change"
    assert proof["legal_name_effective_date"] == "2018-02-21"
    assert proof["marketplace_effective_date"] == "2018-02-27"
    assert proof["new_common_stock_cusip"] == "09857L108"
    assert proof["original_common_stock_cusip"] is None and proof["ticker_event_bracket_start"] is None
    assert proof["outstanding_certificates_remain_valid"] is True
    for field in ("certificate_exchange_required", "mandatory_share_exchange", "historical_interval_interpolated",
                  "provider_identifiers_rewritten", "provider_tickers_rewritten", "capture_time_is_historical_availability",
                  "query_clock_is_exchange_calendar_qualification", "continuous_identity_qualified",
                  "historical_issue_identity_qualified", "economic_conversion_qualified", "economic_accounting_qualified",
                  "economic_inputs_qualified", "tradability_qualified", "point_in_time_qualified", "native_catalog_activated"):
        assert proof[field] is False, field
    assert proof["acquisition_route_only"] is True and proof["training_ready"] is False
    assert route.to_dict()["schema"] == proof["schema"] == SCHEMA
    restored = SecondDocumentedAliasRoute.from_dict(json.loads(json.dumps(route.to_dict())))
    assert restored == route and restored.validate(query) == proof
    assert restored.validate(_query(clock="15:00:00")) == proof
    assert query.ticker == "PCLN" and "/ticker/PCLN/" in query.url
    assert _inventory(originals) == before


@pytest.mark.parametrize("field,value", [
    ("fixed_slot_ticker", "PCLN"), ("fixed_slot_ticker", "META"),
    ("provider_ticker", "BKNG"), ("modern_observation_date", "2026-08-26"),
    ("session_date", "2017-01-04"),
])
def test_route_labels_cannot_expand_reviewed_mapping(originals, field, value):
    with pytest.raises(ValueError):
        replace(_route(originals), **{field: value}).validate(_query())


@pytest.mark.parametrize("day,clock,seconds", [
    ("2017-01-02", "09:30:00", 3600), ("2017-01-04", "09:30:00", 3600),
    ("2018-02-27", "09:30:00", 3600), ("2017-01-03", "09:29:59", 2),
    ("2017-01-03", "15:59:59", 2), ("2017-01-03", "09:30:00", 3601),
    ("2017-01-03", "23:59:59", 2),
])
def test_only_exact_reviewed_day_and_bounded_session_clock(originals, day, clock, seconds):
    route = replace(_route(originals), session_date=day)
    with pytest.raises(ValueError):
        route.validate(_query(day, clock, seconds))


@pytest.mark.parametrize("ticker", ["BKNG", "META", "FB"])
def test_source_query_cannot_be_rewritten_to_fixed_or_unrelated_symbol(originals, ticker):
    with pytest.raises(ValueError):
        _route(originals).validate(replace(_query(), ticker=ticker))


@pytest.mark.parametrize("value", [True, 1483453800000.0, 1483453800001])
def test_unsafe_query_object_cannot_bypass_integral_seconds(originals, value):
    query = _query()
    object.__setattr__(query, "start_ms", value)
    with pytest.raises(ValueError):
        _route(originals).validate(query)


@pytest.mark.parametrize("field,value", [
    ("ticker", "BKNG"), ("name", "Unrelated common shares"), ("cik", "0001326801"),
    ("type", "PFD"), ("primary_exchange", "XNYS"), ("currency_name", "eur"),
    ("market", "crypto"), ("locale", "ca"), ("active", False),
    ("composite_figi", "BBG000BLBVN4"), ("share_class_figi", "BBG001S89N72"),
    ("composite_figi", None), ("share_class_figi", None),
])
def test_rehashed_historical_chain_cannot_override_issue_or_fill_absent_figis(originals, tmp_path, field, value):
    route = _rehash_historical(_copied_route(tmp_path, originals), updates={field: value})
    with pytest.raises(ValueError):
        route.validate(_query())


@pytest.mark.parametrize("case", ["duplicate", "empty"])
def test_rehashed_issuer_population_does_not_choose_a_target_or_infer_absence(originals, tmp_path, case):
    route = _rehash_historical(_copied_route(tmp_path, originals),
                              duplicate=case == "duplicate", empty=case == "empty")
    with pytest.raises(ValueError):
        route.validate(_query())


@pytest.mark.parametrize("field,value", [
    ("ticker", "PCLN"), ("cik", "0001326801"), ("type", "ETF"),
    ("composite_figi", None), ("share_class_figi", "OTHER"), ("primary_exchange", "XNYS"),
])
def test_rehashed_modern_reference_must_not_change_intended_class(originals, tmp_path, field, value):
    route = _rehash_modern(_copied_route(tmp_path, originals), {field: value})
    with pytest.raises(ValueError):
        route.validate(_query())


@pytest.mark.parametrize("case", ["changed-html", "keywords", "report", "changed-receipt",
                                   "wrong-hash", "wrong-kind", "missing", "symlink", "hardlink"])
def test_primary_source_is_exact_original_not_an_arbitrary_rehashed_authority(originals, tmp_path, case):
    route = _copied_route(tmp_path, originals)
    source = route.sec_source
    path = Path(source.path)
    if case in ("changed-html", "keywords", "report"):
        body = {"changed-html": path.read_bytes() + b" ",
                "keywords": b"PCLN BKNG February 21 2018 certificates remain valid 09857L108",
                "report": b'{"approved":true,"same_issue":true}'}[case]
        path.write_bytes(body)
        source = replace(source, sha256=sha256(body).hexdigest())
    elif case == "changed-receipt":
        path = Path(source.receipt_path)
        body = path.read_bytes() + b" "
        path.write_bytes(body)
        source = replace(source, receipt_sha256=sha256(body).hexdigest())
    elif case == "wrong-hash":
        source = replace(source, sha256="0" * 64)
    elif case == "wrong-kind":
        source = replace(source, kind="unreviewed-source")
    elif case == "missing":
        path.unlink()
    else:
        linked = tmp_path / "linked.html"
        if case == "symlink":
            linked.symlink_to(path)
        else:
            linked.hardlink_to(path)
        source = replace(source, path=str(linked))
    with pytest.raises((ValueError, OSError)):
        replace(route, sec_source=source).validate(_query())


@pytest.mark.parametrize("case", ["order", "issue", "boolean-count"])
def test_rehashed_intended_report_cannot_change_fixed_population(originals, tmp_path, case):
    route = _copied_route(tmp_path, originals)
    path = Path(route.intended_issue_source)
    report = json.loads(gzip.decompress(path.read_bytes()))
    if case == "order":
        report["ordered_tickers"] = list(reversed(report["ordered_tickers"]))
    elif case == "issue":
        row = next(r for r in report["security_resolutions"] if r["qt200_ticker"] == "BKNG")
        row["current_reference_evidence"]["current_composite_figi"] = "OTHER"
    else:
        row = next(r for r in report["security_resolutions"] if r["qt200_ticker"] == "BKNG")
        row["current_reference_evidence"]["active_exact_reference_count"] = True
    body = gzip.compress(transport.canonical(report), mtime=0)
    path.write_bytes(body)
    with pytest.raises(ValueError):
        replace(route, intended_issue_sha256=sha256(body).hexdigest()).validate(_query())


def test_original_404_and_single_event_do_not_become_generic_alias_authorities(originals):
    route = _route(originals)
    wrapper = json.loads((originals / "bkng-404-20170103.json").read_bytes())
    assert wrapper["ticker"] == "BKNG" and wrapper["http_status"] == 404 and wrapper["results"] is None
    missing = AliasIdentityRef("dated-ticker-wrapper", str(originals / "bkng-404-20170103.json"),
                               _ORIGINALS["bkng-404-20170103.json"])
    for changed in (replace(route, identity=missing), replace(route, modern_identity=missing,
                    modern_observation_date="2017-01-03")):
        with pytest.raises(ValueError):
            changed.validate(_query())
    event = AliasEventRef(str(originals / "ticker-events-v1.json"), _ORIGINALS["ticker-events-v1.json"], 636)
    row = json.loads(Path(event.path).read_bytes())["rows"][636]
    payload = json.loads(base64.b64decode(row["raw_response_body_base64"], validate=True))
    assert payload["results"]["events"] == [
        {"date": "2018-02-27", "type": "ticker_change", "ticker_change": {"ticker": "BKNG"}}]
    with pytest.raises(ValueError):
        event.interval(fixed_slot="BKNG", provider_ticker="PCLN",
                       issue=("BBG000BLBVN4", "BBG001S89N72", "0001075531", "2018-02-27"))
    with pytest.raises(ValueError):
        SecondAliasRoute("BKNG", "PCLN", "2017-01-03", route.identity, event).validate(_query())


@pytest.mark.parametrize("case", ["extra", "missing", "missing-schema", "wrong-schema",
                                   "untyped-identity", "untyped-modern", "untyped-primary"])
def test_serialization_has_no_free_qualification_or_untyped_reference_override(originals, case):
    route = _route(originals)
    if case in ("extra", "missing", "missing-schema", "wrong-schema"):
        value = route.to_dict()
        if case == "extra":
            value["continuous_identity_qualified"] = True
        elif case == "missing":
            value.pop("identity")
        elif case == "missing-schema":
            value.pop("schema")
        else:
            value["schema"] = "rl-quant.raw-second-exact-day-alias-v1"
        with pytest.raises((TypeError, ValueError, KeyError)):
            SecondDocumentedAliasRoute.from_dict(value)
    else:
        field = {"untyped-identity": "identity", "untyped-modern": "modern_identity",
                 "untyped-primary": "sec_source"}[case]
        with pytest.raises(ValueError):
            replace(route, **{field: route.to_dict()[field]}).validate(_query())


@pytest.mark.parametrize("field", ["identity", "modern_identity", "sec_source", "receipt", "intended"])
def test_original_reference_spelling_cannot_silently_normalize_paths(originals, field):
    route = _route(originals)
    if field == "intended":
        parent, name = route.intended_issue_source.rsplit("/", 1)
        route = replace(route, intended_issue_source=parent + "/./" + name)
    elif field == "receipt":
        parent, name = route.sec_source.receipt_path.rsplit("/", 1)
        route = replace(route, sec_source=replace(route.sec_source, receipt_path=parent + "/./" + name))
    else:
        reference = getattr(route, field)
        parent, name = reference.path.rsplit("/", 1)
        route = replace(route, **{field: replace(reference, path=parent + "/./" + name)})
    with pytest.raises(ValueError):
        route.validate(_query())


def _relocated(tmp_path, originals):
    original = tmp_path / "original"
    route = _copied_route(original, originals)
    # Test-only location-envelope anchors; these are not a materialized market
    # capture, corpus completion, or source-qualification authority.
    capture = original / "capture"
    capture.mkdir()
    plan = transport.canonical({"test_only_route_location_envelope": route.to_dict()})
    completion = transport.canonical({"test_only": True, "training_ready": False})
    (capture / "plan.json").write_bytes(plan)
    (capture / "COMPLETE.json").write_bytes(completion)
    package = tmp_path / "package"
    package.mkdir()
    mapping = []
    for path in sorted(original.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(original).as_posix()
        name = relative if relative.startswith("capture/") else "evidence/" + relative
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        body = path.read_bytes()
        target.write_bytes(body)
        mapping.append(dict(original_path=str(path), published_path=name,
                            bytes=len(body), sha256=sha256(body).hexdigest()))
    map_body = transport.canonical(dict(files=mapping, native_path_resolution_authorized=False, **_FLAGS))
    (package / MAP_NAME).write_bytes(map_body)
    rows = [dict(path=p.relative_to(package).as_posix(), bytes=p.stat().st_size,
                 sha256=sha256(p.read_bytes()).hexdigest()) for p in sorted(package.rglob("*")) if p.is_file()]
    body = transport.canonical(dict(schema="rl-quant.raw-second-evidence-package-v1", files=rows, **_FLAGS))
    (package / "inventory.json").write_bytes(body)
    resolver = EvidenceRelocation(str(package), sha256(body).hexdigest(), sha256(map_body).hexdigest(),
                                   sha256(plan).hexdigest(), sha256(completion).hexdigest())
    original.rename(tmp_path / "retained-original")
    return route, resolver


def test_explicit_relocation_preserves_original_refs_and_never_falls_back(originals, tmp_path, monkeypatch):
    route, resolver = _relocated(tmp_path, originals)
    serialized = route.to_dict()
    monkeypatch.setattr(transport, "_fetch", _denied)
    monkeypatch.setattr(transport.request, "build_opener", _denied)
    before = _inventory(tmp_path)
    with pytest.raises(OSError):
        route.validate(_query())
    verification = resolver.verify(capture_root=Path(resolver.package_root) / "capture",
        plan_sha256=resolver.plan_sha256, completion_sha256=resolver.completion_sha256)
    assert verification["exact_bytes_verified"] is True and verification["training_ready"] is False
    proof = route.validate(_query(), evidence_relocation=resolver)
    restored = SecondDocumentedAliasRoute.from_dict(json.loads(json.dumps(serialized)))
    assert restored.validate(_query(), evidence_relocation=resolver) == proof
    assert route.to_dict() == serialized and proof["provider_ticker"] == "PCLN"
    assert _inventory(tmp_path) == before
    with pytest.raises(ValueError):
        route.validate(_query(), evidence_relocation=object())


@pytest.mark.parametrize("case", ["body", "map", "inventory", "wrong-plan", "wrong-completion"])
def test_explicit_relocation_cannot_repair_source_or_external_pin_tampering(originals, tmp_path, case):
    route, resolver = _relocated(tmp_path, originals)
    package = Path(resolver.package_root)
    if case in ("body", "map", "inventory"):
        name = {"body": "evidence/bkng-sec-20180227.html", "map": MAP_NAME,
                "inventory": "inventory.json"}[case]
        (package / name).write_bytes(b"{}")
    else:
        key = "plan_sha256" if case == "wrong-plan" else "completion_sha256"
        resolver = replace(resolver, **{key: "0" * 64})
    with pytest.raises(ValueError):
        route.validate(_query(), evidence_relocation=resolver)


def test_invalid_explicit_relocation_never_falls_back_to_existing_originals(originals, tmp_path):
    route, resolver = _relocated(tmp_path, originals)
    (tmp_path / "retained-original").rename(tmp_path / "original")
    assert route.validate(_query())["provider_ticker"] == "PCLN"
    before = _inventory(tmp_path)
    with pytest.raises(ValueError):
        route.validate(_query(), evidence_relocation=replace(resolver, inventory_sha256="0" * 64))
    assert _inventory(tmp_path) == before
