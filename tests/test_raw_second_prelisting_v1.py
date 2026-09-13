"""LSF-only, original-byte prelisting disposition acceptance.

QT200_PRELISTING_EVIDENCE_ROOT must contain the 34 originals named below.
Missing originals are errors, not skips; no fabricated trusted-source hashes.
Only negative cases copy/mutate evidence. There is no provider request, market
tensor, optimizer, accounting conversion, or profitability assertion here.
"""

from dataclasses import replace
from datetime import date, datetime, timedelta
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import torch

from rl_quant.data_sources.massive import qt200_research_capture_v1 as transport
from rl_quant.data_sources.massive.raw_second_alias_v1 import AliasIdentityRef
from rl_quant.data_sources.massive.raw_second_prelisting_v1 import (
    ReviewedPrelistingSourceRef, SecondPrelistingDisposition,
)
from rl_quant.datasets.massive_raw_seconds_v1 import SecondQuery

pytestmark = pytest.mark.lsf_gpu
_EASTERN = ZoneInfo("America/New_York")
_INTENDED_SHA = "9f0ef1e22cc939a09302a6bb8970687a7e4e8fee697db9d590c9e710fe0194c0"
_PUBLIC = {
    "pltr-sec-20201231": (
        "847a864869bca3e87c1206269ba27a28b07b4623413e8a38c957f4db67a0c61d",
        "a583bba599df995aa1ceb066855cf6aaa4e319af852fc6705ab3ed58662cfeeb"),
    "crwd-sec-securities-20200131": (
        "ddce5cf4377d5a676d8df0a5e56531899af151986d7b3a2951e4fed1357cdd2d",
        "ad2812480b882c60bc9c0570098bf4452dfdf60a41adcbb6fc10736f26c9023c"),
    "ddog-sec-20191231": (
        "db82c79f1200f78b263481280a6677040a26ea17cbe0fa7a8c63f8a9b9fe8671",
        "1fcb26ca9f185d87f8604cf13f514b7ac3ac8435295b6212e7c117c7fbd3519e"),
    "snow-sec-20210131": (
        "af2c1b07631445594cebded417bd70eb4bf18385a23e7237b5a5ba69fa123750",
        "03337ab59f904cae89d110fe07b1287c8534933f5920f27d7e4e5abad4d175bb"),
    "roku-sec-20171231": (
        "feec44cde1bfa63f20dd00f54486ae6a21e622ef35a3aaddb598518936412747",
        "8ce30a615957c861893b5906cab69f71d45c69a1177a013cc4902cebc107e929"),
    "roku-nasdaq-20170928": (
        "2871c6c7b841b06568e86f3c76b8add68b228bff93e5ba663451cb7ff26bca1f",
        "834dc0ce36184ada351aa39f678a8df388cb6440ed0d3b06fc9600e88a3076c9"),
    "pins-sec-20191231": (
        "1eb0145981183938f8026cea3c8d037c7eb036e1a3112a5eb78d6cfa899e97b1",
        "31f27f2bca9020b9a5ec276b97c03d930ed791955caf1dc635dab0f49244bc98"),
    "abnb-sec-20201231": (
        "38195ebef82f64d93c3f1def64bbeea8a147acbc9e5aa66ca3d0438bd55d735f",
        "db7faffc59162d14abcbfc141ae615f74010f4ecb2712984bdb1948c9e124c37"),
    "coin-sec-20211231": (
        "f08f550ef143b2e9b6fb34b822071d9054727853efab763c95baed36cb36b977",
        "aa01cb33bb860444e00f9718c33da1a87169c1af2dfa9b3b2ddc14bef23ff853"),
    "dell-sec-fy2017": (
        "faa8204b51330f0aa8b6d011d4f9acacef36299f10d63a2e8c15a315580285f2",
        "c261efd09d1042d32b8b8b0e3ffd3824463bbc1f234ece72253404376d077556"),
    "dell-sec-fy2019": (
        "c2e9bcf64b3a9a291bcea501f83b8eaba0e147834692fb0b115ae7b8a10b8d34",
        "459e34fa054eca388b53d00f9eefe71e047b1a1a205f399036c7b1f35153508d"),
}
# Boundary, issuer CIK, original 2022 provider wrapper hash, original sources.
_CASES = {
    "PLTR": ("2020-09-30", "0001321655",
        "9a64e999f6588f0c0c68920d3e5d3481b0e22a3a4b5349bc2791a4e471c765a7", ("pltr-sec-20201231",)),
    "CRWD": ("2019-06-12", "0001535527",
        "1a711222febfe60a09f088324fb0a85b6807287330231dd00f4276280cd4553a", ("crwd-sec-securities-20200131",)),
    "DDOG": ("2019-09-19", "0001561550",
        "1254c15ecccb7782f064adc036264ab67eba2c1ff8d763ebbb889485f914bd02", ("ddog-sec-20191231",)),
    "SNOW": ("2020-09-16", "0001640147",
        "38bfc3331ad47db9154e4bf0416523aff459c1d31a6ff8e9ef57502219ace866", ("snow-sec-20210131",)),
    "ROKU": ("2017-09-28", "0001428439",
        "669cd981f45b074b62eabbe069e7da42a2e14eb1115a4dfead4abec38ecb224f",
        ("roku-sec-20171231", "roku-nasdaq-20170928")),
    "PINS": ("2019-04-18", "0001506293",
        "cf7c3f8fd43cd9654d952a9e871d9e69fcaa336a7f0a7b87dd71a4b724fc47d4", ("pins-sec-20191231",)),
    "ABNB": ("2020-12-10", "0001559720",
        "7f76fbfa1443243b1a1c16f7617b68ee114343585ada0888cdbf7462ddc20393", ("abnb-sec-20201231",)),
    "COIN": ("2021-04-14", "0001679788",
        "c673395d465155021f74ec16e33f850603824f500622b3215dfbc4281e82982e", ("coin-sec-20211231",)),
    "DELL": ("2018-12-26", "0001571996",
        "dba313ec84042a83020156f6059d9b40e6981a9394f7632442d98f3f26e2028f",
        ("dell-sec-fy2017", "dell-sec-fy2019")),
}
_INTERVALS = tuple(ticker for ticker in _CASES if ticker != "DELL")
_NEGATIVE = {
    "snow-wrong-20170103.json": "04d37daf18c18efde0ed807db61c06fc81a30278f4f90e468959b98daa3d0e61",
    "pltr-404-20170103.json": "3b6efe1cec48578deb731f5cf13c0f4de7a5b4655d9d477f5bb672af40430ff4",
}


@pytest.fixture(autouse=True, scope="module")
def assigned_gpu():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1


@pytest.fixture(scope="module")
def originals():
    root = Path(os.environ["QT200_PRELISTING_EVIDENCE_ROOT"])
    expected = {"intended-issues.json.gz": _INTENDED_SHA, **_NEGATIVE}
    for kind, hashes in _PUBLIC.items():
        expected[kind + ".html"] = hashes[0]
        expected[kind + ".receipt.json"] = hashes[1]
    for ticker, (_, _, digest, _) in _CASES.items():
        expected[ticker.lower() + "-intended-20220103.json"] = digest
    assert len(expected) == 34
    for name, digest in expected.items():
        assert sha256(transport.read_regular(root / name, 8 * 1024 * 1024)).hexdigest() == digest, name
    return root


def _disposition(root, ticker="SNOW"):
    boundary, _, digest, kinds = _CASES[ticker]
    return SecondPrelistingDisposition(
        fixed_slot_ticker=ticker, session_date="2017-01-03", public_trading_start_date=boundary,
        identity=AliasIdentityRef("dated-ticker-wrapper", str(root / (ticker.lower() + "-intended-20220103.json")), digest),
        identity_observation_date="2022-01-03", intended_issue_source=str(root / "intended-issues.json.gz"),
        intended_issue_sha256=_INTENDED_SHA,
        primary_sources=tuple(ReviewedPrelistingSourceRef(kind, str(root / (kind + ".html")), _PUBLIC[kind][0],
                             str(root / (kind + ".receipt.json")), _PUBLIC[kind][1]) for kind in kinds),
    )


def _query(ticker="SNOW", day="2017-01-03", clock="09:30:00", seconds=3600):
    start = int(datetime.fromisoformat(day + "T" + clock).replace(tzinfo=_EASTERN).timestamp()) * 1000
    return SecondQuery(ticker, start, start + (seconds - 1) * 1000)


@pytest.mark.parametrize("ticker", tuple(_CASES))
def test_actual_evidence_preserves_slot_without_fabricating_a_capture(originals, ticker, monkeypatch):
    disposition = _disposition(originals, ticker)
    sources = [Path(disposition.identity.path), Path(disposition.intended_issue_source)]
    sources += [Path(path) for source in disposition.primary_sources for path in (source.path, source.receipt_path)]
    before = {str(path): sha256(path.read_bytes()).hexdigest() for path in sources}

    def denied(*args, **kwargs):
        raise AssertionError("Prelisting validation cannot issue a provider request")

    monkeypatch.setattr(transport, "_fetch", denied)
    query = _query(ticker)
    proof = disposition.validate(query)
    intended = json.loads(gzip.decompress((originals / "intended-issues.json.gz").read_bytes()))
    wrapper = json.loads(Path(disposition.identity.path).read_bytes())
    assert proof["fixed_slot_index"] == intended["ordered_tickers"].index(ticker)
    assert proof["fixed_axis_size"] == len(intended["ordered_tickers"]) == 200
    assert proof["ordered_tickers_sha256"] == "92002983b9b6686bf8b6f5d6f88d212cf97f61f1f98bb55d49f262fa33b491fe"
    assert proof["query"] == {"ticker": ticker, "start_ms": query.start_ms, "end_ms": query.end_ms}
    assert proof["intended_cik"] == _CASES[ticker][1]
    assert proof["intended_composite_figi"] == wrapper["results"]["composite_figi"]
    assert proof["intended_share_class_figi"] == wrapper["results"]["share_class_figi"]
    assert proof["intended_security_class"] == ("Class C common stock" if ticker == "DELL" else "Class A common stock")
    assert proof["reason"] == "intended-class-before-public-trading"
    assert proof["market_observation_status"] == "known-unavailable"
    assert proof["unavailable_end_date_exclusive"] == ("2017-01-04" if ticker == "DELL" else _CASES[ticker][0])
    assert proof["fixed_axis_slot_preserved"] and proof["unavailable_mask_required"]
    assert proof["provider_capture_reference"] is None and proof["boundary_intraday_timestamp_ms"] is None
    for field in ("acquisition_required", "known_empty_interval", "synthetic_provider_response_created",
                  "synthetic_market_records_created", "market_tensor_constructed", "no_same_ticker_records_claimed",
                  "private_security_nonexistence_claimed", "all_historical_listing_statuses_qualified",
                  "point_in_time_qualified", "first_trading_day_intraday_qualified",
                  "query_clock_is_exchange_calendar_qualification", "continuous_identity_qualified",
                  "economic_accounting_qualified", "tradability_qualified", "native_catalog_activated",
                  "training_authorized", "training_ready", "tracking_stock_alias_authorized"):
        assert proof[field] is False, field
    restored = SecondPrelistingDisposition.from_dict(json.loads(json.dumps(disposition.to_dict())))
    assert restored == disposition and restored.validate(query) == proof
    assert {str(path): sha256(path.read_bytes()).hexdigest() for path in sources} == before


@pytest.mark.parametrize("ticker", _INTERVALS)
def test_cutoff_is_exclusive_and_cannot_hide_the_first_public_trading_day(originals, ticker):
    disposition = _disposition(originals, ticker)
    boundary = _CASES[ticker][0]
    previous = (date.fromisoformat(boundary) - timedelta(days=1)).isoformat()
    proof = replace(disposition, session_date=previous).validate(_query(ticker, previous))
    assert proof["unavailable_end_date_exclusive"] == boundary
    for day in (boundary, (date.fromisoformat(boundary) + timedelta(days=1)).isoformat()):
        with pytest.raises(ValueError):
            replace(disposition, session_date=day).validate(_query(ticker, day))
    with pytest.raises(ValueError):
        replace(disposition, public_trading_start_date=previous).validate(_query(ticker))


def test_roku_conflict_is_preserved_and_both_original_sources_are_required(originals):
    disposition = _disposition(originals, "ROKU")
    proof = disposition.validate(_query("ROKU"))
    assert proof["boundary_conflicting_source_dates"] == ["2017-09-28", "2017-09-29"]
    assert proof["boundary_policy"] == "conservative-earliest-source-date-with-Nasdaq-corroboration"
    assert proof["boundary_intraday_timestamp_ms"] is None
    for sources in (disposition.primary_sources[:1], disposition.primary_sources[1:],
                    tuple(reversed(disposition.primary_sources)), disposition.primary_sources * 2):
        with pytest.raises(ValueError):
            replace(disposition, primary_sources=sources).validate(_query("ROKU"))


def test_pltr_exchange_change_and_snow_unadorned_name_do_not_rewrite_provider_identity(originals):
    pltr = _disposition(originals, "PLTR").validate(_query("PLTR"))
    snow = _disposition(originals).validate(_query())
    assert pltr["intended_cik"] == "0001321655" and pltr["continuous_identity_qualified"] is False
    assert snow["intended_provider_name"] == "Snowflake Inc."
    assert snow["intended_security_class"] == "Class A common stock"


def test_dell_class_c_is_exact_session_only_not_a_dvmt_alias_or_interpolated_interval(originals):
    disposition = _disposition(originals, "DELL")
    proof = disposition.validate(_query("DELL"))
    assert proof["intended_security_class"] == "Class C common stock"
    assert proof["reviewed_session_dates"] == ["2017-01-03"]
    assert proof["disposition_scope"] == "exact-reviewed-session"
    assert proof["unavailable_end_date_exclusive"] == "2017-01-04"
    assert proof["public_trading_start_date_is_exclusion_interval_boundary"] is False
    assert proof["later_trading_dates_are_corroboration_only"] is True
    assert proof["public_trading_conventions"] == {"when_issued": "2018-12-26", "regular_way": "2018-12-28"}
    # The original provider list_date differs; it is retained, never adopted
    # as a first trade or an interpolated no-market interval.
    assert proof["provider_list_date"] == "2018-12-19"
    for day in ("2017-01-04", "2017-02-03", "2018-12-25", "2018-12-26", "2018-12-28"):
        with pytest.raises(ValueError):
            replace(disposition, session_date=day).validate(_query("DELL", day))
    with pytest.raises(ValueError):
        disposition.validate(_query("DVMT"))
    for sources in (disposition.primary_sources[:1], disposition.primary_sources[1:]):
        with pytest.raises(ValueError):
            replace(disposition, primary_sources=sources).validate(_query("DELL"))


@pytest.mark.parametrize("ticker,name", [("SNOW", "snow-wrong-20170103.json"), ("PLTR", "pltr-404-20170103.json")])
def test_wrong_issuer_or_404_is_never_an_affirmative_prelisting_authority(originals, ticker, name):
    disposition = _disposition(originals, ticker)
    wrapper = json.loads((originals / name).read_bytes())
    if ticker == "SNOW":
        assert wrapper["http_status"] == 200 and wrapper["results"]["ticker"] == "SNOW"
        assert wrapper["results"]["name"] == "Intrawest Resorts Holdings, Inc."
        assert "cik" not in wrapper["results"]
    else:
        assert wrapper["http_status"] == 404 and wrapper["results"] is None
    identity = AliasIdentityRef("dated-ticker-wrapper", str(originals / name), _NEGATIVE[name])
    with pytest.raises(ValueError):
        replace(disposition, identity=identity, identity_observation_date="2017-01-03").validate(_query(ticker))
    with pytest.raises(ValueError):
        replace(disposition, primary_sources=()).validate(_query(ticker))


@pytest.mark.parametrize("case", ["altered-body", "keyword-proof", "report-proof", "altered-receipt",
                                   "wrong-receipt", "missing-original", "symlink", "hardlink"])
def test_primary_originals_and_receipts_cannot_be_replaced_even_with_new_matching_hashes(originals, tmp_path, case):
    disposition = _disposition(originals)
    source = disposition.primary_sources[0]
    if case in ("altered-body", "keyword-proof", "report-proof"):
        original = Path(source.path).read_bytes()
        raw = {"altered-body": original + b" ", "keyword-proof": b"SNOW Class A September 16 2020 no public market",
               "report-proof": b'{"approved":true,"start":"2020-09-16"}'}[case]
        assert raw != original
        target = tmp_path / "replaced.html"
        target.write_bytes(raw)
        source = replace(source, path=str(target), sha256=sha256(raw).hexdigest())
    elif case == "altered-receipt":
        row = json.loads(Path(source.receipt_path).read_bytes())
        row["http_status"] = 404
        raw = json.dumps(row).encode()
        target = tmp_path / "replaced.receipt.json"
        target.write_bytes(raw)
        source = replace(source, receipt_path=str(target), receipt_sha256=sha256(raw).hexdigest())
    elif case == "wrong-receipt":
        other = _disposition(originals, "COIN").primary_sources[0]
        source = replace(source, receipt_path=other.receipt_path, receipt_sha256=other.receipt_sha256)
    elif case == "missing-original":
        source = replace(source, path=str(tmp_path / "absent.html"))
    else:
        # Never hardlink an immutable original: link only two fresh test copies.
        copied = tmp_path / "copy.html"
        copied.write_bytes(Path(source.path).read_bytes())
        linked = tmp_path / "linked.html"
        if case == "symlink":
            linked.symlink_to(copied)
        else:
            linked.hardlink_to(copied)
        source = replace(source, path=str(linked))
    with pytest.raises((ValueError, OSError)):
        replace(disposition, primary_sources=(source,)).validate(_query())


@pytest.mark.parametrize("case", ["different-slot", "unreviewed-slot", "different-provider-ticker",
                                   "wrong-date", "pre-scope", "pre-open", "post-close", "over-hour",
                                   "cross-date", "wrong-observation-date", "source-list", "other-source-kind"])
def test_disposition_does_not_expand_the_reviewed_issue_or_interval(originals, case):
    disposition, query = _disposition(originals), _query()
    if case == "different-slot":
        disposition = replace(disposition, fixed_slot_ticker="COIN")
    elif case == "unreviewed-slot":
        disposition = replace(disposition, fixed_slot_ticker="AAPL")
    elif case == "different-provider-ticker":
        query = replace(query, ticker="FB")
    elif case == "wrong-date":
        disposition = replace(disposition, session_date="2017-01-04")
    elif case == "pre-scope":
        disposition, query = replace(disposition, session_date="2017-01-02"), _query(day="2017-01-02")
    elif case == "pre-open":
        query = _query(clock="09:29:59", seconds=2)
    elif case == "post-close":
        query = _query(clock="15:59:59", seconds=2)
    elif case == "over-hour":
        query = _query(seconds=3601)
    elif case == "cross-date":
        query = _query(clock="23:59:59", seconds=2)
    elif case == "wrong-observation-date":
        disposition = replace(disposition, identity_observation_date="2022-01-04")
    elif case == "source-list":
        disposition = replace(disposition, primary_sources=list(disposition.primary_sources))
    else:
        disposition = replace(disposition, primary_sources=_disposition(originals, "COIN").primary_sources)
    with pytest.raises(ValueError):
        disposition.validate(query)


@pytest.mark.parametrize("case", ["issuer", "share-class", "axis-order", "missing-slot", "ready-label"])
def test_rehashed_intended_population_cannot_change_the_exclusion_axis_or_issue(originals, tmp_path, case):
    disposition = _disposition(originals)
    evidence = json.loads(gzip.decompress(Path(disposition.intended_issue_source).read_bytes()))
    target = next(row for row in evidence["security_resolutions"] if row["qt200_ticker"] == "SNOW")
    if case == "issuer":
        target["current_reference_evidence"]["current_issuer_cik"] = "0000000000"
    elif case == "share-class":
        target["current_reference_evidence"]["current_share_class_figi"] = "other-class"
    elif case == "axis-order":
        evidence["ordered_tickers"] = list(reversed(evidence["ordered_tickers"]))
    elif case == "missing-slot":
        evidence["ordered_tickers"].remove("SNOW")
    else:
        evidence["training_ready_for_adaptive_v5"] = True
    raw = gzip.compress(json.dumps(evidence).encode(), mtime=0)
    path = tmp_path / "replaced-issues.json.gz"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        replace(disposition, intended_issue_source=str(path), intended_issue_sha256=sha256(raw).hexdigest()).validate(_query())


@pytest.mark.parametrize("case", ["extra-field", "missing-field", "untyped-identity", "untyped-source",
                                   "extra-source-field", "missing-receipt", "tuple-not-json-list"])
def test_serialization_requires_the_typed_complete_evidence_graph(originals, case):
    row = _disposition(originals).to_dict()
    if case == "extra-field":
        row["training_ready"] = True
    elif case == "missing-field":
        del row["session_date"]
    elif case == "untyped-identity":
        row["identity"] = "SNOW"
    elif case == "untyped-source":
        row["primary_sources"] = ["approved"]
    elif case == "extra-source-field":
        row["primary_sources"][0]["approved"] = True
    elif case == "missing-receipt":
        del row["primary_sources"][0]["receipt_sha256"]
    else:
        row["primary_sources"] = tuple(row["primary_sources"])
    with pytest.raises(ValueError):
        SecondPrelistingDisposition.from_dict(row)


def test_changed_original_is_reopened_after_an_initial_success(originals, tmp_path):
    disposition = _disposition(originals)
    source = disposition.primary_sources[0]
    path = tmp_path / "original-copy.html"
    path.write_bytes(Path(source.path).read_bytes())
    disposition = replace(disposition, primary_sources=(replace(source, path=str(path)),))
    assert disposition.validate(_query())["acquisition_required"] is False
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        disposition.validate(_query())
