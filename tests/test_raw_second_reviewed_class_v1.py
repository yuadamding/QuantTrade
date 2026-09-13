"""LSF-only exact-original class-route acceptance; never fetch provider data.

QT200_REVIEWED_CLASS_EVIDENCE_ROOT supplies unchanged original wrappers,
public bodies/receipts and the intended population. Missing evidence fails.
Negative tests mutate only owned fixture copies; no source pins are changed.
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

from rl_quant.data_sources.massive import qt200_research_capture_v1 as io
from rl_quant.data_sources.massive import raw_second_reviewed_class_v1 as classes
from rl_quant.data_sources.massive.raw_second_alias_v1 import AliasIdentityRef, validate_fixed_slot_identity
from rl_quant.data_sources.massive.raw_second_evidence_v1 import EvidenceRelocation, MAP_NAME
from rl_quant.datasets.massive_raw_seconds_v1 import SecondQuery

pytestmark = pytest.mark.lsf_gpu
_TICKERS = ("HLT", "RCL", "CCL")
_DATES = ("2017-01-03", "2022-01-03", "2026-08-26")
_EASTERN = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True, scope="module")
def assigned_gpu():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1


def _files(ticker):
    spec = classes._CLASSES[ticker]
    return (tuple(f"{ticker.lower()}-{day}.json" for day in _DATES)
            + tuple(name + suffix for name in spec["sources"] for suffix in (".html", ".receipt.json"))
            + ("intended-issues.json.gz",))


@pytest.fixture(scope="module")
def originals():
    root = Path(os.environ["QT200_REVIEWED_CLASS_EVIDENCE_ROOT"])
    assert root.is_absolute() and not root.is_symlink()
    expected = {"intended-issues.json.gz": classes._INTENDED_SHA}
    for ticker in _TICKERS:
        expected.update(zip((f"{ticker.lower()}-{day}.json" for day in _DATES),
                            classes._CLASSES[ticker]["wrappers"], strict=True))
    for kind, (_, _, body, _, receipt) in classes._PUBLIC.items():
        expected[kind + ".html"], expected[kind + ".receipt.json"] = body, receipt
    assert len(expected) == 28
    assert {p.name for p in root.iterdir()} == set(expected)
    for name, digest in expected.items():
        assert sha256(io.read_regular(root / name, 8 * 1024**2)).hexdigest() == digest, name
    return root


def _query(ticker, day="2017-01-03", clock="09:30:00", seconds=3600):
    start = int(datetime.fromisoformat(day + "T" + clock).replace(tzinfo=_EASTERN).timestamp()) * 1000
    return SecondQuery(ticker, start, start + (seconds - 1) * 1000)


def _route(root, ticker):
    spec = classes._CLASSES[ticker]
    identities = tuple(AliasIdentityRef("dated-ticker-wrapper", str(root / f"{ticker.lower()}-{day}.json"), h)
                       for day, h in zip(_DATES, spec["wrappers"], strict=True))
    refs = tuple(classes.ReviewedClassSourceRef(kind, str(root / (kind + ".html")), classes._PUBLIC[kind][2],
        str(root / (kind + ".receipt.json")), classes._PUBLIC[kind][4]) for kind in spec["sources"])
    return classes.SecondReviewedClassRoute(ticker, ticker, _DATES[0], *identities,
        str(root / "intended-issues.json.gz"), classes._INTENDED_SHA, refs)


def _copy(root, originals, ticker):
    root.mkdir(parents=True, exist_ok=False)
    for name in _files(ticker):
        (root / name).write_bytes((originals / name).read_bytes())
    return _route(root, ticker)


def _inventory(root):
    return {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _denied(*args, **kwargs):
    raise AssertionError("Source validation/replay cannot issue a provider request")


@pytest.mark.parametrize("ticker", _TICKERS)
def test_exact_originals_preserve_axis_figi_absence_and_nontraining_scope(originals, ticker, monkeypatch):
    monkeypatch.setattr(io, "_fetch", _denied)
    monkeypatch.setattr(io.request, "build_opener", _denied)
    before = _inventory(originals)
    route = _route(originals, ticker)
    proof = route.validate(_query(ticker))
    assert proof["fixed_slot_index"] == {"HLT": 59, "RCL": 60, "CCL": 61}[ticker]
    assert proof["provider_ticker"] == proof["fixed_slot_ticker"] == ticker
    assert proof["ordered_tickers_sha256"] == "92002983b9b6686bf8b6f5d6f88d212cf97f61f1f98bb55d49f262fa33b491fe"
    expected_absence = {"HLT": [True, False, False], "RCL": [True, True, True], "CCL": [False, False, True]}[ticker]
    assert [r["figi_fields_absent"] for r in proof["provider_observations"]] == expected_absence
    assert proof["intended_observation"]["current_issue_identifier_observed"] is (ticker == "HLT")
    assert proof["proof_basis"] == "reviewed-exact-original-class-evidence-not-nullable-identifier-equality"
    for key in ("historical_interval_interpolated", "provider_identifiers_rewritten", "provider_tickers_rewritten",
                "raw_values_transformed", "capture_time_is_historical_availability",
                "query_clock_is_exchange_calendar_qualification", "continuous_identity_qualified",
                "historical_issue_identity_qualified", "economic_conversion_qualified", "economic_accounting_qualified",
                "economic_inputs_qualified", "tradability_qualified", "point_in_time_qualified",
                "native_catalog_activated", "training_authorized", "training_ready"):
        assert proof[key] is False, key
    assert proof["acquisition_route_only"] is True
    restored = classes.SecondReviewedClassRoute.from_dict(json.loads(json.dumps(route.to_dict())))
    assert restored == route and restored.validate(_query(ticker)) == proof
    assert route.validate(_query(ticker, clock="15:00:00")) == proof
    assert _inventory(originals) == before


def test_three_relationships_are_not_a_generic_same_cik_override(originals):
    hlt, rcl, ccl = (_route(originals, t).validate(_query(t)) for t in _TICKERS)
    h = hlt["relationship_details"]
    assert h["excluded_provider_symbols"] == ["HLT WI", "PK", "HGV"]
    assert h["reverse_split_boundary"] == "after-market-close-2017-01-03"
    assert h["fractional_cash_and_distribution_ledger_implemented"] is False
    r = rcl["relationship_details"]
    assert r["legal_registrant"] == "Royal Caribbean Cruises Ltd."
    assert r["jurisdiction"] == "Republic of Liberia" and r["par_value_usd"] == "0.01"
    assert r["synthetic_issue_identifier"] is None and r["preferred_stock_is_distinct"] is True
    c = ccl["relationship_details"]
    assert (c["original_public_cusip"], c["successor_public_cusip"]) == ("143658300", "G2004J103")
    assert c["historical_class"] == "old-CCL-common-stock-paired-stock"
    assert c["CUK_is_historical_alias"] is False and c["intraday_conversion_timestamp_ms"] is None
    assert len({p["relationship"] for p in (hlt, rcl, ccl)}) == 3


@pytest.mark.parametrize("ticker", _TICKERS)
def test_legacy_nullable_identifier_gate_is_not_weakened(originals, ticker):
    route = _route(originals, ticker)
    with pytest.raises(ValueError):
        validate_fixed_slot_identity(fixed_slot_ticker=ticker, provider_ticker=ticker, session_date=_DATES[0],
            identity=route.identity, intended_issue_source=Path(route.intended_issue_source),
            intended_issue_sha256=route.intended_issue_sha256)


@pytest.mark.parametrize("ticker", _TICKERS)
@pytest.mark.parametrize("day,clock,seconds", [
    ("2017-01-04", "09:30:00", 3600), ("2017-01-02", "09:30:00", 3600),
    ("2017-01-03", "09:29:59", 2), ("2017-01-03", "15:59:59", 2),
    ("2017-01-03", "09:30:00", 3601), ("2026-05-07", "09:30:00", 3600),
])
def test_no_date_interpolation_overnight_or_event_boundary_expansion(originals, ticker, day, clock, seconds):
    with pytest.raises(ValueError):
        replace(_route(originals, ticker), session_date=day).validate(_query(ticker, day, clock, seconds))


@pytest.mark.parametrize("ticker", _TICKERS)
@pytest.mark.parametrize("field,value", [("fixed_slot_ticker", "AAPL"), ("provider_ticker", "CUK"),
                                        ("session_date", "2017-01-04"), ("provider_ticker", "HLT WI")])
def test_no_substituted_slot_or_provider_symbol(originals, ticker, field, value):
    with pytest.raises(ValueError):
        replace(_route(originals, ticker), **{field: value}).validate(_query(ticker))


@pytest.mark.parametrize("value", [True, 1483453800000.0, 1483453800001])
def test_unsafe_query_cannot_bypass_integral_second_contract(originals, value):
    query = _query("HLT")
    object.__setattr__(query, "start_ms", value)
    with pytest.raises(ValueError):
        _route(originals, "HLT").validate(query)


@pytest.mark.parametrize("ticker", _TICKERS)
@pytest.mark.parametrize("reference", ["identity", "modern_identity", "latest_identity"])
@pytest.mark.parametrize("mutation", ["issuer", "class", "synthetic-figi"])
def test_rehashed_original_wrapper_cannot_change_class_or_backfill_ids(originals, tmp_path, ticker, reference, mutation):
    route = _copy(tmp_path / "evidence", originals, ticker)
    ref = getattr(route, reference)
    path = Path(ref.path)
    wrapper = json.loads(path.read_bytes())
    payload = json.loads(base64.b64decode(wrapper["raw_response_body_base64"], validate=True))
    payload["results"].update({"cik": "0000000001"} if mutation == "issuer" else
                              {"type": "PFD"} if mutation == "class" else {"composite_figi": "SYNTHETIC"})
    raw = io.canonical(payload)
    wrapper.update(results=payload["results"], raw_response_body_base64=base64.b64encode(raw).decode(),
                   raw_response_body_sha256=sha256(raw).hexdigest(), raw_response_content_length=len(raw))
    body = io.canonical(wrapper)
    path.write_bytes(body)
    changed = replace(route, **{reference: replace(ref, sha256=sha256(body).hexdigest())})
    with pytest.raises(ValueError):
        changed.validate(_query(ticker))


@pytest.mark.parametrize("ticker", _TICKERS)
@pytest.mark.parametrize("part", ["body", "receipt", "population", "intended", "reordered"])
def test_rehashed_public_or_intended_replacements_do_not_authorize(originals, tmp_path, ticker, part):
    route = _copy(tmp_path / "evidence", originals, ticker)
    if part in ("body", "receipt"):
        ref = route.primary_sources[0]
        path = Path(ref.path if part == "body" else ref.receipt_path)
        path.write_bytes(b"caller-authored replacement")
        field = "sha256" if part == "body" else "receipt_sha256"
        ref = replace(ref, **{field: sha256(path.read_bytes()).hexdigest()})
        route = replace(route, primary_sources=(ref, *route.primary_sources[1:]))
    elif part == "population":
        route = replace(route, primary_sources=route.primary_sources[:-1])
    elif part == "reordered":
        route = replace(route, primary_sources=tuple(reversed(route.primary_sources)))
    else:
        path = Path(route.intended_issue_source)
        evidence = json.loads(gzip.decompress(path.read_bytes()))
        row = evidence["security_resolutions"][classes._CLASSES[ticker]["slot"]]["current_reference_evidence"]
        row.update(current_issue_identifier_observed=True, current_composite_figi="SYNTHETIC")
        body = gzip.compress(io.canonical(evidence), mtime=0)
        path.write_bytes(body)
        route = replace(route, intended_issue_sha256=sha256(body).hexdigest())
    with pytest.raises(ValueError):
        route.validate(_query(ticker))


@pytest.mark.parametrize("case", ["schema", "extra", "missing", "untyped", "nested-extra"])
def test_strict_serialized_route_rejects_caller_extension(originals, case):
    row = _route(originals, "RCL").to_dict()
    if case == "schema":
        row["schema"] = "generic-cik-override"
    elif case == "extra":
        row["training_ready"] = True
    elif case == "missing":
        del row["latest_identity"]
    elif case == "untyped":
        row["primary_sources"] = tuple(row["primary_sources"])
    else:
        row["identity"]["ignore_figi"] = True
    with pytest.raises(ValueError):
        classes.SecondReviewedClassRoute.from_dict(row)


@pytest.mark.parametrize("ticker", _TICKERS)
def test_explicit_byte_relocation_preserves_original_references(originals, tmp_path, ticker, monkeypatch):
    route = _route(originals, ticker)
    proof = route.validate(_query(ticker))
    target = tmp_path / "published"
    target.mkdir()
    files, mapping = [], []
    for name in _files(ticker):
        body = (originals / name).read_bytes()
        (target / name).write_bytes(body)
        files.append(dict(path=name, bytes=len(body), sha256=sha256(body).hexdigest()))
        mapping.append(dict(original_path=str(originals / name), published_path=name,
                            bytes=len(body), sha256=sha256(body).hexdigest()))
    # Explicit test-only location anchors: not a real market capture or a
    # qualification terminal. Original reference strings are unchanged.
    capture = target / "capture"
    capture.mkdir()
    anchors = {}
    for name in ("plan.json", "COMPLETE.json"):
        body = io.canonical(dict(test_only_location_envelope=True, training_ready=False, name=name))
        (capture / name).write_bytes(body)
        anchors[name] = sha256(body).hexdigest()
        files.append(dict(path="capture/" + name, bytes=len(body), sha256=anchors[name]))
        mapping.append(dict(original_path=str(tmp_path / "original-capture" / name),
                            published_path="capture/" + name, bytes=len(body), sha256=anchors[name]))
    flags = dict(training_ready=False, training_authorized=False, remote_native_replay_qualified=False,
                 native_catalog_activated=False, original_references_rewritten=False)
    descriptor = dict(files=mapping, native_path_resolution_authorized=False, **flags)
    (target / MAP_NAME).parent.mkdir()
    (target / MAP_NAME).write_bytes(io.canonical(descriptor))
    map_sha = sha256((target / MAP_NAME).read_bytes()).hexdigest()
    files.append(dict(path=MAP_NAME, bytes=(target / MAP_NAME).stat().st_size,
                      sha256=sha256((target / MAP_NAME).read_bytes()).hexdigest()))
    inventory = io.canonical(dict(schema="rl-quant.raw-second-evidence-package-v1",
                                  files=sorted(files, key=lambda r: r["path"]), **flags))
    (target / "inventory.json").write_bytes(inventory)
    resolver = EvidenceRelocation(str(target), sha256(inventory).hexdigest(), map_sha,
                                   anchors["plan.json"], anchors["COMPLETE.json"])
    original_read = io.read_regular

    def no_original(path, cap):
        assert not Path(path).is_relative_to(originals)
        return original_read(path, cap)

    monkeypatch.setattr(io, "read_regular", no_original)
    monkeypatch.setattr(io, "_fetch", _denied)
    verified = resolver.verify(capture_root=capture, plan_sha256=anchors["plan.json"],
                               completion_sha256=anchors["COMPLETE.json"])
    assert verified["exact_bytes_verified"] is True and verified["training_ready"] is False
    assert route.validate(_query(ticker), evidence_relocation=resolver) == proof
